import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers.models.bert.modeling_bert import BertConfig, BertModel
from .bert_model import BertCrossLayer
from . import heads, objectives, meter_utils
from .clip_model import build_model, adapt_position_encoding
from transformers import RobertaConfig, RobertaModel
import torch.distributed as dist
import copy
from flm.utils.utils import adapt_vocab_size
from flm.modules.flm_tools import get_corr_bi_attention_mask


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args["world_size"])]
        dist.all_gather(output, tensor)
        ctx.rank = args["rank"]
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size *
                        ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


class FLMTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        try:
            self.save_hyperparameters(config)
        except:
            pass
        self.hparams.config = config

        self.is_vit = ('swin' not in config['vit'])
        # self.is_mae = 'mae' in config['vit']

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_top_layer"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
                is_decoder=config["is_causal_mask"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_top_layer"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
                is_decoder=config["is_causal_mask"],
            )

        resolution_after = config['image_size']

        self.all_gather = AllGather_multi.apply
        self.cross_modal_text_transform = nn.Linear(
            config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(
            config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings_flm = nn.Embedding(
            2, config["hidden_size_for_fusion"])
        self.token_type_embeddings_flm.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                build_model(config['vit'], resolution_after=resolution_after)
                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])

            torch.distributed.barrier()

        self.vit_model = build_model(
            config['vit'], resolution_after=resolution_after)
        self.causal_mask = config['is_causal_mask']

        if config["text_encoder_from_scratch"]:
            te_config = RobertaConfig.from_pretrained(config['tokenizer'])
            if self.causal_mask:
                te_config.is_decoder = True
            self.text_transformer = BertModel(config=te_config)
            # text_transformer_hidden_dim = te_config.hidden_size

        elif 'roberta' in config['tokenizer']:
            te_config = RobertaConfig.from_pretrained(config['tokenizer'])
            if self.causal_mask:
                te_config.is_decoder = True
            self.text_transformer = RobertaModel.from_pretrained(
                config['tokenizer'], config=te_config)
            self.text_transformer.encoder.layer = nn.ModuleList(
                [self.text_transformer.encoder.layer[_]
                    for _ in range(config['num_bottom_layer'])])
        else:
            te_config = BertModel.from_pretrained(config['tokenizer'])
            if self.causal_mask:
                te_config.is_decoder = True
            self.text_transformer = BertModel.from_pretrained(
                config['tokenizer'], config=te_config)

            if True:
                self.text_transformer.encoder.layer = nn.ModuleList(
                    [self.text_transformer.encoder.layer[_]
                        for _ in range(config['num_bottom_layer'])])

        vocab_size = config["vocab_size"]

        if config['add_new_bos_token']:
            print('add two additional tokens')
            vocab_size = config["vocab_size"] + 2
            self.text_transformer.resize_token_embeddings(vocab_size)
            bert_config.vocab_size = vocab_size

        self.cross_modal_text_layers = nn.ModuleList(
            [BertCrossLayer(bert_config)
                for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)
        self.cross_modal_layers = self.cross_modal_text_layers

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["ar"] > 0:
            self.lm_score = heads.MLMHead(bert_config)
            self.lm_score.apply(objectives.init_weights)

        if config["flm_backbone"]:
            self.text_transformer2 = self.text_transformer
            self.cross_modal_text_layers_r = self.cross_modal_text_layers
            # self.cross_modal_text_pooler_r = self.cross_modal_text_pooler
            self.cross_modal_text_transform_r = self.cross_modal_text_transform
            self.cross_modal_text_transform_f = self.cross_modal_text_transform

            self.cross_modal_text_pooler_f = heads.Pooler(
                config["hidden_size_for_fusion"])
            self.cross_modal_text_pooler_f.apply(objectives.init_weights)

            self.fusion_token_embedding = self.text_transformer.embeddings
            bert_config_fusion = copy.deepcopy(bert_config)
            bert_config_fusion.hidden_size = config['hidden_size_for_fusion']
            bert_config_fusion.num_attention_heads = config['num_heads_fusion']
            self.fusion_layers_top = nn.ModuleList([BertCrossLayer(
                bert_config_fusion)
                for _ in range(config['num_reconstructor_top_layer'])])
            self.fusion_layers_bottom = nn.ModuleList(
                [BertCrossLayer(bert_config_fusion)
                    for _ in range(config['num_reconstructor_bottom_layer'])])

            if True:  # remove unused params in self-attention layers
                for layer in self.fusion_layers_top:
                    layer.attention = None
                for layer in self.fusion_layers_bottom:
                    layer.attention = None

            self.lm_type_embeddings = nn.Embedding(2, config["hidden_size"])

            if config["loss_names"]["flm"] > 0:
                self.lm_score = heads.MLMHead(bert_config_fusion)
                self.lm_score.apply(objectives.init_weights)
                if config['share_lm_scorer_weights']:
                    self.lm_score_r = self.lm_score
                    self.lm_score_f = self.lm_score
                else:
                    self.lm_score_r = heads.MLMHead(bert_config_fusion)
                    self.lm_score_r.apply(objectives.init_weights)
                    self.lm_score_f = heads.MLMHead(bert_config_fusion)
                    self.lm_score_f.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score_flm = heads.ITMHead(
                config["hidden_size_for_fusion"])
            self.itm_score.apply(objectives.init_weights)
            self.itm_score_flm.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"] if not config['flm_backbone'] \
            else self.hparams.config["hidden_size_for_fusion"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(
                self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]

            state_dict = adapt_position_encoding(
                state_dict, after=resolution_after,
                patch_size=self.hparams.config['patch_size'])
            state_dict = adapt_vocab_size(state_dict, vocab_size)

            if True:
                r = self.load_state_dict(state_dict, strict=False)
                print(' Missing keys in loading pretrained model: {},\
                    Unexpected keys number: {}'.format(
                    (r.missing_keys), (r.unexpected_keys)))

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(
                3, config['hidden_size'])  # TODO
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        meter_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and \
                self.hparams.config["test_only"]:
            ckpt = torch.load(
                self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(
                state_dict, after=resolution_after,
                patch_size=self.hparams.config['patch_size'])
            state_dict = adapt_vocab_size(state_dict, vocab_size)
            r = self.load_state_dict(state_dict, strict=False)
            print(' Missing keys in loading pretrained model: {}, \
                Unexpected keys number: {}'.format(
                (r.missing_keys), (r.unexpected_keys)))
        self.config = config

    def infer(self, *args, **kargs):
        if 'flm_backbone' in kargs:
            is_flm_backbone = kargs.pop('flm_backbone')
        else:
            is_flm_backbone = self.config['flm_backbone']

        if is_flm_backbone:
            return self.infer_three_stream(*args, **kargs)
        else:
            return self.infer_one_stream(*args, **kargs)

    def get_extended_attention_mask(self, attention_mask, input_shape, device, is_decoder=False):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(
                    batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - \
                        causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length,
                                 prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None,
                                                      :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def infer_one_stream(
        self,
        batch,
        mask_text=False,
        do_lm=False,
        image_token_type_idx=1,
        img=None,
        return_intermediate=False,
        image_only=False,
        text_only=False,
        enable_causal_mask=None,
        txt_key="text",
        keep_image_token_embed=False,
    ):
        is_decoder = self.causal_mask if enable_causal_mask is None else enable_causal_mask
        if not text_only:
            if True:
                if img is None:
                    if f"image_{image_token_type_idx - 1}" in batch:
                        imgkey = f"image_{image_token_type_idx - 1}"
                    else:
                        imgkey = "image"
                    img = batch[imgkey][0]

                raw_image_embeds = self.vit_model(img)
        if image_only:
            return {"image_embeds": raw_image_embeds}

        input_suffix = "_mlm" if mask_text else ""
        text_ids = batch[f"{txt_key}_ids{input_suffix}"]
        output_suffix = "_lm" if do_lm else input_suffix
        text_labels = batch[f"{txt_key}_labels{output_suffix}"]
        text_masks = batch[f"{txt_key}_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.get_extended_attention_mask(
            text_masks, input_shape, device, is_decoder)
        # if is_decoder and self.causal_mask_w_post_cls:
        if is_decoder:
            extend_text_masks = torch.cat(
                (extend_text_masks[:, :, -1:], extend_text_masks[:, :, 1:]), dim=2)
            extend_text_masks[:, :, 1:, 0] = -10000.

        num_bottom_layer = self.config['num_bottom_layer']
        for layer in self.text_transformer.encoder.layer[:num_bottom_layer]:
            text_embeds = layer(
                text_embeds, attention_mask=extend_text_masks)[0]
        raw_text_embeds = text_embeds

        if text_only:
            return {"text_embeds": raw_text_embeds}

        if return_intermediate:
            ret = {
                "text_embeds": raw_text_embeds,
                "text_mask": extend_text_masks,
                "image_embeds": raw_image_embeds,
                "image_feats": raw_image_embeds,
                'img': img,
            }
            return ret

        # Cross-Modal Fusion
        text_embeds = self.cross_modal_text_transform(raw_text_embeds)
        text_embeds = text_embeds + \
            self.token_type_embeddings(
                torch.zeros_like(text_embeds[..., 0]).long())
        image_embeds = self.cross_modal_image_transform(raw_image_embeds)

        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(
            1)), dtype=torch.long, device=image_embeds.device)
        extend_image_masks = image_masks.reshape(
            image_masks.size(0), 1, 1, image_masks.size(1))
        extend_image_masks = (1.0 - extend_image_masks) * -10000.0

        if keep_image_token_embed:
            image_embeds = image_embeds + self.token_type_embeddings(
                torch.full(image_embeds.shape[:2], 1, device=image_embeds.device))
        else:
            image_embeds = image_embeds + self.token_type_embeddings(torch.full(
                image_embeds.shape[:2], image_token_type_idx, device=image_embeds.device))
        x, y = text_embeds, image_embeds

        for text_layer in self.cross_modal_text_layers:
            x1 = text_layer(x, y, attention_mask=extend_text_masks,
                            encoder_attention_mask=extend_image_masks)
            x = x1[0]

        text_feats, image_feats = x, y
        cls_feats = self.cross_modal_text_pooler(x)
        cls_feats = torch.cat([cls_feats, cls_feats], dim=-1)

        ret = {
            "text_embeds": raw_text_embeds,
            "image_embeds": raw_image_embeds,
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            'img': img,
        }

        return ret

    def get_img_text_merged_mask(self, text_mask, image_mask, is_causal=False, pad_token_id=0):
        b, hn, w1, h1 = text_mask.shape
        w2, h2 = image_mask.shape[-2:]
        if w1 == 1 and w1 != h1:
            text_mask = text_mask.expand([-1, -1, h1, -1])
        if w2 == 1 and w2 != h2:
            image_mask = image_mask.expand([-1, -1, h2, -1])
        top_pad = torch.ones(
            (b, hn, h1, h2), device=text_mask.device) * pad_token_id
        down_pad = torch.ones(
            (b, hn, h2, h1), device=image_mask.device) * pad_token_id
        if is_causal:
            top_pad = torch.ones(
                (b, hn, h1, w2), device=text_mask.device) * (-10000.)
        top = torch.cat([text_mask, top_pad], dim=-1)
        down = torch.cat([down_pad, image_mask], dim=-1)
        mask = torch.cat([top, down], dim=-2)
        return mask

    def infer_three_stream(
        self,
        batch,
        mask_text=False,
        do_lm=False,
        image_token_type_idx=1,
        img=None,
        return_intermediate=False,
        image_only=False,
        text_only=False,
        enable_causal_mask=None,
        txt_key='text',
        keep_image_token_embed=False
    ):

        assert mask_text is False
        do_lm = True

        if not text_only:
            if img is None:
                if f"image_{image_token_type_idx - 1}" in batch:
                    imgkey = f"image_{image_token_type_idx - 1}"
                else:
                    imgkey = "image"
                img = batch[imgkey][0]

            image_only_embeds = self.vit_model(img)

        if image_only:
            return {"image_embeds": image_only_embeds}

        input_suffix = "_mlm" if mask_text else ""
        text_ids = batch[f"{txt_key}_ids{input_suffix}"]
        output_suffix = "_lm" if do_lm else input_suffix
        text_labels = batch[f"{txt_key}_labels{output_suffix}"]
        text_masks = batch[f"{txt_key}_masks"]
        fusion_ids = batch[f"{txt_key}_all_masks_ids"]
        if self.config.get('only_use_cls_for_flm', False):
            fusion_ids = fusion_ids[:, :1]
        flm_masks = batch[f"{txt_key}_flm_masks"].unsqueeze(dim=1)

        text_embeds_f = self.fusion_token_embedding(input_ids=fusion_ids)
        # if hasattr(self, 'dim_expand_flag_bottom') and self.dim_expand_flag_bottom:
        #     text_embeds_f = self.query_to_fusion_dim(text_embeds_f)

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        text_embeds_r = self.text_transformer2.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(
            text_masks, input_shape, device)
        if (not self.training and self.config['full_att_mask_for_eval']) or self.config['full_att_mask']:
            extend_text_masks = torch.ones_like(
                text_masks)[:, None, :, None] * text_masks[:, None, None, :]
        nonpad_area = (text_masks.unsqueeze(
            2) * text_masks.unsqueeze(1)).unsqueeze(1)
        nonpad_area[:, :, 0] = 0
        nonpad_area[:, :, :, 0] = 0

        if self.causal_mask:
            extend_text_masks = torch.cat(
                (extend_text_masks[:, :, -1:], extend_text_masks[:, :, 1:]), dim=2)
            extend_text_masks[:, :, 1:, 0] = -10000.

            if (not self.training and self.config['full_att_mask_for_eval']) or self.config['full_att_mask']:
                extend_text_masks_r = extend_text_masks

            elif self.training and self.config["random_flm_mask"]:
                extend_text_masks = flm_masks * nonpad_area + \
                    (1-nonpad_area) * extend_text_masks
                extend_text_masks_r = flm_masks.transpose(
                    2, 3) * nonpad_area + (1-nonpad_area) * flm_masks
            else:
                extend_text_masks_r = extend_text_masks.transpose(
                    2, 3) * nonpad_area + (1-nonpad_area) * extend_text_masks

            if (not self.training and self.config['full_att_mask_for_eval']) or self.config['full_att_mask']:
                bi_attention_mask = torch.cat(
                    (extend_text_masks, extend_text_masks_r), dim=-1)
            else:
                if self.config['span_corruption_rate'] > 0:
                    bi_attention_mask = get_corr_bi_attention_mask(
                        extend_text_masks, extend_text_masks_r, self.config['span_corruption_rate'])
                else:
                    bi_attention_mask = self.get_bi_attention_mask(
                        extend_text_masks, extend_text_masks_r)
                if self.config.get('only_use_cls_for_flm', False):
                    bi_attention_mask = bi_attention_mask[:, :, :1, :]

        num_bottom_layer = self.config['num_bottom_layer']
        assert self.config['num_reconstructor_bottom_layer'] <= num_bottom_layer
        for i in range(num_bottom_layer):
            text_embeds = self.text_transformer.encoder.layer[i](
                text_embeds, attention_mask=extend_text_masks)[0]
            text_embeds_r = self.text_transformer2.encoder.layer[i](
                text_embeds_r, attention_mask=extend_text_masks_r)[0]
            t_num_layers = num_bottom_layer - \
                self.config['num_reconstructor_bottom_layer']
            if i >= t_num_layers:
                bi_contexts = torch.cat((text_embeds, text_embeds_r), dim=1)
                text_embeds_f = self.fusion_layers_bottom[i-t_num_layers](
                    text_embeds_f, bi_contexts, attention_mask=None,
                    encoder_attention_mask=bi_attention_mask,
                    disable_self_attention=True)[0]

        if self.config['num_reconstructor_bottom_layer'] > 0:
            text_only_embeds = text_embeds_f
        else:
            text_only_embeds = text_embeds + text_embeds_r

        if text_only:
            return {"text_embeds": text_only_embeds}

        if return_intermediate:
            ret = {
                "text_embeds": text_only_embeds,
                "image_embeds": image_only_embeds,
                "text_mask": text_masks,
            }
            return ret

        text_embeds = self.cross_modal_text_transform(text_embeds)
        text_embeds_r = self.cross_modal_text_transform_r(text_embeds_r)
        text_embeds_f = self.cross_modal_text_transform_f(text_embeds_f)
        text_embeds = text_embeds + \
            self.token_type_embeddings(torch.zeros_like(text_masks))

        text_embeds_r = text_embeds_r + \
            self.token_type_embeddings_flm(torch.zeros_like(text_masks))
        text_embeds_f = text_embeds_f + \
            self.token_type_embeddings_flm(torch.ones_like(fusion_ids))

        image_embeds = self.cross_modal_image_transform(image_only_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(
            1)), dtype=torch.long, device=image_embeds.device)
        extend_image_masks = image_masks.reshape(
            image_masks.size(0), 1, 1, image_masks.size(1))
        if keep_image_token_embed:
            image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, 1))
        else:
            image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx))

        x, y = text_embeds, image_embeds
        x_r, x_f = text_embeds_r, text_embeds_f

        num_top_layer = self.config['num_top_layer']
        for i in range(num_top_layer):
            x = self.cross_modal_text_layers[i](
                x, y,
                attention_mask=extend_text_masks,
                encoder_attention_mask=extend_image_masks)[0]
            x_r = self.cross_modal_text_layers_r[i](
                x_r, y,
                attention_mask=extend_text_masks_r,
                encoder_attention_mask=extend_image_masks)[0]
            t_only_num_layer = num_top_layer - \
                self.config['num_reconstructor_top_layer']
            if i >= t_only_num_layer:
                bi_contexts = torch.cat([x, x_r], dim=1)
                x_f = self.fusion_layers_top[i-t_only_num_layer](
                    x_f, bi_contexts,
                    attention_mask=None,
                    encoder_attention_mask=bi_attention_mask,
                    disable_self_attention=True)[0]

        text_feats, image_feats = x_f, y
        text_feats1 = x
        text_feats2 = x_r

        cls_feats = self.cross_modal_text_pooler_f(x_f)

        ret = {
            "text_embeds": text_only_embeds,
            "image_embeds": image_only_embeds,
            "text_feats": text_feats,
            "text_feats1": text_feats1,
            "text_feats2": text_feats2,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks
        }

        return ret

    def get_bi_attention_mask(self, mask, mask_r):
        N = mask.shape[-1]
        bi_mask = torch.cat([mask, mask_r], dim=-1)
        bi_mask[:, :, torch.arange(1, N), torch.arange(1, N)] = -10000.
        bi_mask[:, :, torch.arange(1, N), N + torch.arange(1, N)] = -10000.
        return bi_mask

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ss_backbone = True
            enable_causal_mask = False
            ret.update(objectives.compute_mlm(
                self, batch, ss_backbone, enable_causal_mask))

        # Language Modeling
        if "ar" in self.current_tasks:
            ret.update(objectives.compute_lm(self, batch))

        # Language Modeling
        if "flm" in self.current_tasks:
            ret.update(objectives.compute_lm(self, batch, lm_type='flm'))

        # Image Text Matching
        if "itm" in self.current_tasks:
            enable_causal_mask = False
            ss_backbone = True
            ret.update(objectives.compute_itm(
                self, batch, ss_backbone, enable_causal_mask))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        if self.config['debug']:
            print('train step: ', self.current_epoch,
                  self.global_step, 'batch_idx: ', batch_idx)
        if batch['is_sep_mlm']:
            total_loss = 0
            for _batch in batch['batch']:
                total_loss += self.sub_training_step(_batch, batch_idx)
            total_loss = total_loss / len(batch['batch'])
            return total_loss
        else:
            return self.sub_training_step(batch, batch_idx)

    def sub_training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        loss_weights = self.hparams.config["loss_names"]
        # pdb.set_trace()
        total_loss = sum([loss_weights[k.split('_')[0]] *
                         v for k, v in output.items() if "loss" in k])
        if self.config['debug']:
            # import pdb
            # pdb.set_trace()
            print('   ', [(k, v) for k, v in output.items() if "loss" in k])
            print('   total_loss: {}'.format(total_loss))
        return total_loss

    def training_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        if self.config['debug']:
            print('val step: ', self.current_epoch,
                  self.global_step, 'batch_idx: ', batch_idx)
        meter_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        if not self.hparams.config['skip_test_step']:
            output = self(batch)
        ret = dict()
        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(
                outs, model_name, self.config['exp_path'])
        meter_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return meter_utils.set_schedule(self)
