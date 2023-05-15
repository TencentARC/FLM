from random import shuffle
import torch
import functools
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    # DataCollatorForWholeWordMask,
    BertTokenizer,
    RobertaTokenizer,
)

from flm.utils.whole_word_masking import DataCollatorForWholeWordMask


class text_preprocessor():
    """prepend or append special tokens"""

    def __init__(self, config) -> None:
        self.prepend_bos = config['add_new_bos_token'] and config['prepend_bos_token']
        self.append_eos = config['add_new_bos_token'] and config['append_eos_token']

    def __call__(self, text):
        text = text.rstrip().rstrip('.').rstrip() + '.'
        if self.prepend_bos:
            text = '<bos>' + ' ' + text
        if self.append_eos:
            text = text + ' ' + '<eos>'
        return text


def flm_collator(attention_mask, mask_ratio, disable_shuffle=True, label_strategy='none'):
    """get flm masks and labels"""
    text_len = attention_mask.sum(1)
    bs, max_len = attention_mask.size()
    flm_masks = -10000. * torch.ones(bs, max_len, max_len)
    # attention_mask.unsqueeze(dim=2) * attention_mask.unsqueeze(dim=1)
    flm_random_ids = []
    mask_num = torch.distributions.Binomial(
        text_len.float() - 1, mask_ratio).sample().int()
    for i in range(len(text_len)):
        flm_random_id = torch.randperm(text_len[i] - 1) + 1
        flm_random_id = flm_random_id[:text_len[i] - 1 - mask_num[i]]
        if disable_shuffle:
            flm_random_id = torch.sort(flm_random_id)[0]
        flm_random_ids.append(flm_random_id)
        # print(flm_random_id)
        for j in range(len(flm_random_id)):
            if flm_random_id[j] < 0:
                break
            else:
                flm_masks[i,
                          flm_random_id[j:j + 1].repeat(j+1),
                          flm_random_id[:j+1]] = 0

    flm_label = None
    if label_strategy == 'none':
        pass
    else:

        if label_strategy == 'object':
            pass
        elif label_strategy == 'concrete':
            pass
    return flm_random_ids, flm_masks, flm_label


def sep_collator(flatten_encodings, mlm_collator, mask_ratio, pred_corr_ratio) -> None:
    if pred_corr_ratio > 1:
        repeat_num = int(pred_corr_ratio)
        group_mlms = [[] for i in range(repeat_num)]
        mlms = mlm_collator(flatten_encodings)
        # print('mlms', mlms)
        for idx, flatten_encoding in enumerate(flatten_encodings):
            token_num = len(flatten_encoding['attention_mask'])
            chunk_size = token_num // repeat_num + 1
            org_input_id = torch.tensor(flatten_encoding['input_ids'])
            mlm_input_id = mlms['input_ids'][idx]
            mlm_labels = mlms['labels'][idx]
            ava_mask_reg = torch.tensor(flatten_encoding['attention_mask']) * (
                1 - torch.tensor(flatten_encoding['special_tokens_mask']))
            perm = torch.randperm(token_num)
            groups = perm.split(chunk_size)
            assert len(groups) == repeat_num
            for i in range(repeat_num):
                group_mask = torch.zeros(token_num).long()
                group_mask[groups[i]] = 1
                group_input_id = org_input_id * \
                    (1-group_mask) + mlm_input_id * group_mask
                group_label = -100 * torch.ones(token_num).long()
                group_label[group_mask.bool()] = mlm_labels[group_mask.bool()]
                group_mlm = {'input_ids': group_input_id,
                             'labels': group_label}
                group_mlms[i].append(group_mlm)
                # print(group_mask)
        for i in range(repeat_num):
            group_mlms[i] = {'input_ids': torch.stack([_['input_ids'] for _ in group_mlms[i]]),
                             'labels': torch.stack([_['labels'] for _ in group_mlms[i]])}
        return group_mlms

    elif pred_corr_ratio < 1:
        mlms = mlm_collator(flatten_encodings)
        group_labels = []
        # print('mlms', mlms)
        for idx, flatten_encoding in enumerate(flatten_encodings):
            token_num = len(flatten_encoding['attention_mask'])
            mlm_input_id = mlms['input_ids'][idx]
            mlm_labels = mlms['labels'][idx]
            perm = torch.randperm(token_num)[:int(token_num * pred_corr_ratio)]
            group_label = -100 * torch.ones(token_num).long()
            group_label[perm] = mlm_labels[perm]
            group_labels.append(group_label)

        group_mlm = {'input_ids': mlms['input_ids'],
                     'labels': torch.stack(group_labels, dim=0)}
        return group_mlm


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            if 'roberta' in from_pretrained:
                RobertaTokenizer.from_pretrained(from_pretrained)
            else:
                BertTokenizer.from_pretrained(
                    from_pretrained, do_lower_case="uncased" in from_pretrained
                )
        torch.distributed.barrier()

    if 'roberta' in from_pretrained:
        return RobertaTokenizer.from_pretrained(from_pretrained)
    elif 'gpt2' in from_pretrained:
        from transformers import GPT2Tokenizer, GPT2Model
        return GPT2Tokenizer.from_pretrained('gpt2')
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.image_only = _config["image_only"]

        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        if _config['add_new_bos_token']:
            self.tokenizer.add_tokens(['<bos>', '<eos>'])
        self.vocab_size = self.tokenizer.vocab_size

        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = {'mlm_collator':
                             collator(tokenizer=self.tokenizer,
                                      mlm=True,
                                      mlm_probability=_config["mlm_prob"]),
                             "flm_collator":
                             functools.partial(
                                 flm_collator,
                                 mask_ratio=_config["flm_mask_prob"],
                                 disable_shuffle=_config["disable_flm_shuffle"]),
                             }

        self.text_preprocessor = text_preprocessor(_config)
        self.setup_flag = False
        self.max_dataset_len = _config.get('max_dataset_len', -1)

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
            disable_sep_mlm=False,
            text_preprocessor=self.text_preprocessor,
            max_dataset_len=self.max_dataset_len
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
            text_preprocessor=self.text_preprocessor,
            max_dataset_len=self.max_dataset_len
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
                tokenizer=self.tokenizer,
                text_preprocessor=self.text_preprocessor,
                max_dataset_len=self.max_dataset_len
            )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
            tokenizer=self.tokenizer,
            text_preprocessor=self.text_preprocessor,
            max_dataset_len=self.max_dataset_len
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            tokenizer=self.tokenizer,
            text_preprocessor=self.text_preprocessor,
            max_dataset_len=self.max_dataset_len
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
