from ..datasets import ConceptualCaptionDataset
from .datamodule_base import BaseDataModule


# Conceptual Caption 3M datamodule
class ConceptualCaptionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ConceptualCaptionDataset

    @property
    def dataset_name(self):
        return "gcc"
