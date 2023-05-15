# flake8: noqa
from glob import glob
from .base_dataset import BaseDataset
from .conceptual_caption_dataset import ConceptualCaptionDataset
import io
from PIL import Image


# Conceptual Caption 12M Dataset
class ConceptualCaption12mDataset(ConceptualCaptionDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"conceptual_caption12M_train_{i}" for i in range(96)]
        elif split == "val":
            # names = [f"conceptual_caption_val_{i}" for i in range(1)]
            names = []

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
