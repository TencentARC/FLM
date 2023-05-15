from ..datasets import ConceptualCaption12mDataset
from .datamodule_base import BaseDataModule


# Conceptual Caption 12M datamodule
class ConceptualCaption12mDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ConceptualCaption12mDataset

    @property
    def dataset_name(self):
        return "gcc"
