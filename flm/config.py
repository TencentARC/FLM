from sacred import Experiment

ex = Experiment("FLM")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,  # used for pretraining MLM-based models
        "ar": 0,  # used for pretraining AR-based models or finetuning on captioning tasks
        "flm": 0,  # used for pretraining FLM-based models
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    only_use_cls_for_flm = False

    debug = False
    log_path = ""
    is_causal_mask = False

    causal_mask_w_post_cls = False
    get_caption_metric = False
    get_mlm_caption_metric = False
    get_cl_recall_metric = False
    get_cl_itm_recall_metric = False

    skip_test_step = False

    flm_backbone = False
    temperature = 0.05
    random_flm_mask = False
    disable_flm_shuffle = False
    flm_mask_prob = 0.
    text_encoder_from_scratch = False
    full_att_mask_for_eval = False
    full_att_mask = False
    enable_flm_aux_lm_loss = False
    flm_aux_lm_loss_l2r_weight = 1.0
    flm_aux_lm_loss_r2l_weight = 1.0

    span_corruption_rate = 0

    share_lm_scorer_weights = True

    max_dataset_len = -1

    hidden_size_for_fusion = 768

    caption_prompt = None
    add_new_bos_token = False
    prepend_bos_token = False
    append_eos_token = False

    # webdataset
    allow_val_webdataset = False

    # adaptive top bottom layer number for flm
    num_reconstructor_bottom_layer = 6
    num_reconstructor_top_layer = 6
    num_bottom_layer = 6

    # enable_prefix_LM=False
    prefix_lm_alpha = 1.0
    flm_prediction_rate = 1.0

    # exp name
    exp_name = "flm"
    seed = 2022
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    # hloss_weights = _hloss_weights({'lmcl': 0.1})
    # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    batch_size = 4096

    prepare_data_per_node = True
    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    draw_false_image = 1
    image_only = False
    resolution_before = 224

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 50
    tokenizer = ".cache/bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False  # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    hidden_size = 768
    num_heads = 12
    num_heads_fusion = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1
    # truncate_bottom_text_encoder_layer = False

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 0.2
    num_sanity_val_steps = 2
    test_only = False
    ckpt_save_top_k = 1

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 8
    # num_nodes = 1
    load_path = ""
    fix_exp_version = False
    num_workers = 8
    precision = 32


@ex.named_config
def causal_flm():
    is_causal_mask = True
    causal_mask_w_post_cls = True
    flm_backbone = True


@ex.named_config
def causal_lm():
    is_causal_mask = True
    causal_mask_w_post_cls = True


@ex.named_config
def mlm():
    exp_name = "mlm"
    # datasets = ["gcc"]
    loss_names = _loss_names({"mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True


@ex.named_config
def ar():
    exp_name = "ar"
    # datasets = ["gcc"]
    loss_names = _loss_names({"ar": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True


@ex.named_config
def flm():
    exp_name = "flm"
    # datasets = ["gcc"]
    loss_names = _loss_names({"flm": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True

    is_causal_mask = True
    causal_mask_w_post_cls = True
    # disable_cross_modal_image_layer=True
    # cross_modal_layer='text_only'
    flm_backbone = True
    enable_flm_aux_lm_loss = True


@ex.named_config
def flm_itm():
    exp_name = "flm_itm"
    # datasets = ["gcc"]
    loss_names = _loss_names({"flm": 1, "itm": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True
    enable_flm_aux_lm_loss = True


@ex.named_config
def ft_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    max_text_len = 50
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 288


@ex.named_config
def ft_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 0.5
    lr_mult_head = 50
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    max_text_len = 50
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576


@ex.named_config
def ft_irtr_coco():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def ft_cap_coco():
    exp_name = "finetune_caption_coco"

    loss_names = _loss_names({"ar": 0.5})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    get_caption_metric = True
    get_mlm_caption_metric = False
    get_recall_metric = False
    draw_false_text = 0
    learning_rate = 3e-5
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384

    caption_prompt = '<bos>'
    add_new_bos_token = True
    prepend_bos_token = True
    append_eos_token = True
    datasets = ["coco"]
    per_gpu_batchsize = 64


# @ex.named_config
# def add_bos_eos_tokens():
#     add_new_bos_token=True
#     prepend_bos_token=True
#     append_eos_token=True

@ex.named_config
def zs_irtr_coco():
    test_only = True
    skip_test_step = True
    get_recall_metric = True
    get_cl_recall_metric = False

    exp_name = "zs_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def ft_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def ft_cl_itm_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1, "cl": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    get_cl_itm_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def zs_irtr_f30k():
    test_only = True
    skip_test_step = True
    get_recall_metric = True
    get_cl_recall_metric = False

    exp_name = "zeroshot_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def ft_cl_irtr_f30k():
    exp_name = "finetune_cl_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"cl": 1.0})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    get_cl_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def zs_cl_irtr_f30k():
    test_only = True
    skip_test_step = True
    get_recall_metric = False
    get_cl_recall_metric = True

    exp_name = "zs_cl_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"cl": 1.0})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    get_cl_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def zs_cl_irtr_coco():
    test_only = True
    skip_test_step = True
    get_recall_metric = False
    get_cl_recall_metric = True

    exp_name = "zs_cl_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"cl": 0.5})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    get_cl_recall_metric = True
    draw_false_text = 15
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


@ex.named_config
def ft_snli_clip_bert():
    exp_name = "finetune_snli"
    datasets = ["snli"]
    loss_names = _loss_names({"snli": 1})
    batch_size = 64
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-uncased"
    max_text_len = 50
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end

# vision encoder
@ex.named_config
def swin32_base224():
    vit = "swin_base_patch4_window7_224_in22k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 224


@ex.named_config
def swin32_base384():
    vit = "swin_base_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 384


@ex.named_config
def swin32_large384():
    vit = "swin_large_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1536
    resolution_before = 384


@ex.named_config
def clip32():
    vit = 'ViT-B/32'
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


@ex.named_config
def clip16():
    vit = 'ViT-B/16'
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


@ex.named_config
def clip14():
    vit = 'ViT-L/14'
    patch_size = 14
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 1024


@ex.named_config
def clip14_336():
    vit = 'ViT-L/14@336px'
    image_size = 336
    patch_size = 14
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 1024


@ex.named_config
def mae_vit_huge_patch14():
    vit = 'mae_vit_huge_patch14'
    image_size = 224
    patch_size = 14
    train_transform_keys = ["mae"]
    val_transform_keys = ["mae"]


@ex.named_config
def mae_vit_large_patch16():
    vit = 'mae_vit_large_patch16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["mae"]
    val_transform_keys = ["mae"]


@ex.named_config
def mae_vit_base_patch16():
    vit = 'mae_vit_base_patch16'
    image_size = 224
    patch_size = 16
    train_transform_keys = ["mae"]
    val_transform_keys = ["mae"]

# text encoder


@ex.named_config
def text_roberta():
    tokenizer = ".cache/roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768


# @ex.named_config
# def text_clip():
#     tokenizer = ".cache/roberta-base"
#     vocab_size = 50265
#     input_text_embed_size = 768

@ex.named_config
def text_roberta_large():
    tokenizer = ".cache/roberta-large"
    vocab_size = 50265
    input_text_embed_size = 1024


# random augmentation
@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet_randaug"]


@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip_randaug"]


@ex.named_config
def mae_randaug():
    train_transform_keys = ["mae_randaug"]
