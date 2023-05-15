from builtins import hasattr
import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules
import webdataset as wds


# datamodule for mutiple datasets
class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()

        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](
            _config) for key in datamodule_keys}
        self.dms = [v for k, v in self.dm_dicts.items()]

        self.batch_size = self.dms[0].batch_size
        self.vocab_size = self.dms[0].vocab_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist
        self.allow_val_webdataset = _config['allow_val_webdataset']

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        def check_webdataset(dataset):
            if hasattr(dataset, 'inner_dataset'):
                return True

        for dm in self.dms:
            dm.setup(stage)

        if check_webdataset(self.dms[0].train_dataset):
            assert len(
                self.dms) == 1, 'does not support webdataset instance larger than 1'
            self.train_dataset = self.dms[0].train_dataset.inner_dataset
            # self.train_dataset.append(wds.batched(self.batch_size))
        else:
            self.train_dataset = ConcatDataset(
                [dm.train_dataset for dm in self.dms])

        if check_webdataset(self.dms[0].val_dataset) and self.allow_val_webdataset:
            self.val_dataset = self.dms[0].val_dataset.inner_dataset
            # self.val_dataset.append(wds.batched(self.batch_size))
        else:
            self.val_dataset = ConcatDataset(
                [dm.val_dataset for dm in self.dms])

        if check_webdataset(self.dms[0].test_dataset) and self.allow_val_webdataset:
            self.test_dataset = self.dms[0].test_dataset.inner_dataset
            # self.test_dataset.append(wds.batched(self.batch_size))
        else:
            self.test_dataset = ConcatDataset(
                [dm.test_dataset for dm in self.dms])

        self.tokenizer = self.dms[0].tokenizer

        self.train_collate = functools.partial(
            self.dms[0].train_dataset.collate, mlm_collator=self.dms[0].mlm_collator
        )
        self.val_collate = functools.partial(
            self.dms[0].val_dataset.collate, mlm_collator=self.dms[0].mlm_collator
        )
        self.test_collate = functools.partial(
            self.dms[0].test_dataset.collate, mlm_collator=self.dms[0].mlm_collator
        )

        if self.dist:
            if isinstance(self.train_dataset, wds.DataPipeline):
                self.train_sampler = None
            else:
                self.train_sampler = DistributedSampler(
                    self.train_dataset, shuffle=True)
            if isinstance(self.val_dataset, wds.DataPipeline) and self.allow_val_webdataset:
                self.val_sampler = None
            else:
                self.val_sampler = DistributedSampler(
                    self.val_dataset, shuffle=True)
            if isinstance(self.test_dataset, wds.DataPipeline) and self.allow_val_webdataset:
                self.test_sampler = None
            else:
                self.test_sampler = DistributedSampler(
                    self.test_dataset, shuffle=False)

        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.val_collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.test_collate,
        )
        return loader
