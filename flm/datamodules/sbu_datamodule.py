from ..datasets import SBUCaptionDataset
from .datamodule_base import BaseDataModule


# SBU Caption datamodule
class SBUCaptionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SBUCaptionDataset

    @property
    def dataset_name(self):
        return "sbu"
