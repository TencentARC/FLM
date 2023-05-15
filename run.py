import os
import copy
import json
import torch
import pytorch_lightning as pl
from flm.modules import FLMTransformerSS
from flm.datamodules.multitask_datamodule import MTDataModule
from flm.config import ex


def args_checker(config):
    if config['enable_flm_aux_lm_loss']:
        assert config['loss_names']['flm'] > 0
        assert config['flm_backbone']
        assert config['is_causal_mask']
        assert config["hidden_size"] == config["hidden_size_for_fusion"], \
            "only support hidden_size_for_fusion=hidden_size"


@ex.automain
def run(_config):
    config = copy.deepcopy(_config)
    args_checker(config)
    # print(os.environ)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    nnodes = int(os.environ.get('NNODES', 1))
    config["world_size"] = world_size
    config["rank"] = rank
    config["nnodes"] = nnodes
    config["num_nodes"] = nnodes
    config["local_rank"] = local_rank

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    pl.seed_everything(config["seed"])
    dm = MTDataModule(config, dist=True)
    exp_name = f'{config["exp_name"]}'

    os.makedirs(config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=None,  # use logger's path
        save_top_k=config["ckpt_save_top_k"],
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        filename='epoch_{epoch:0>3d}-step_{step:0>6d}-val_score_{val/the_metric:.3f}',
        auto_insert_metric_name=False,
    )

    version = 0 if config['fix_exp_version'] else None

    logger = pl.loggers.TensorBoardLogger(
        config["log_dir"],
        name=f'{exp_name}_seed{config["seed"]}_from_{config["load_path"].split("/")[-1][:-5]}',
        version=version,
    )
    config['exp_path'] = logger.root_dir

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        config["num_gpus"]
        if isinstance(config["num_gpus"], int)
        else len(config["num_gpus"])
    )

    print(config)
    available_batch_size = config["per_gpu_batchsize"] * \
        num_gpus * config["num_nodes"]
    grad_steps = max(config["batch_size"] // (available_batch_size), 1)

    max_steps = config["max_steps"] if config["max_steps"] is not None else None

    if local_rank == 0:
        # print(os.environ)
        print(
            f' Node Num: {num_gpus}, Total GPU Numbers: {num_gpus * config["num_nodes"]}')
        print(
            f' Total Batch Size: {config["batch_size"]}, \
                Available Batch Size: {available_batch_size}, \
                    Per GPU Batch Size: {config["per_gpu_batchsize"]},\
                        Grad Steps: {grad_steps}')
        print(f' Resume_from: {config["resume_from"]}')
        print(f' Load_path: {config["load_path"]}')
        print(' All configs: \n', json.dumps(
            _config, sort_keys=True, indent=4, separators=(',', ':')))

    model = FLMTransformerSS(config)

    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=config["prepare_data_per_node"],
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=100,
        flush_logs_every_n_steps=100,
        resume_from_checkpoint=config["resume_from"],
        weights_summary="top",
        fast_dev_run=config["fast_dev_run"],
        val_check_interval=config["val_check_interval"],
        # progress_bar_refresh_rate= 5 if config['debug'] else 200,
        num_sanity_val_steps=config['num_sanity_val_steps'],
    )

    if not config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
