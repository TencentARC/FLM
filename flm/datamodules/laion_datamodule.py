from ..datasets import LaionDataset
from .datamodule_base import BaseDataModule


# LAION-400M datamodule
class LaionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return LaionDataset

    @property
    def dataset_name(self):
        return "laion"
