from ..datasets import Laion100mDataset
from .datamodule_base import BaseDataModule


# LAION-100M datamodule, a random subset of LAION-400M
class Laion100mDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Laion100mDataset

    @property
    def dataset_name(self):
        return "laion"
