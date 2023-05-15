from .base_webdataset import WebDataset
import io
from PIL import Image


# Laion-400M Dataset
class LaionDataset(WebDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == 'train':
            # location = "/group/30042/public_datasets/LAION-400M/raw/data/38872.tar"
            # location = "/group/30042/public_datasets/LAION-400M/raw/data/{00000..42757}.tar"
            location = "/group/30042/public_datasets/LAION-400M/raw/data/{00001..42757}.tar"
            infinite_loader = True
        elif split == "val":
            location = '/group/30042/public_datasets/LAION-400M/raw/data/00000.tar'
            infinite_loader = False
        elif split == 'test':
            location = '/group/30042/public_datasets/LAION-400M/raw/data/00000.tar'
            infinite_loader = False
        super().__init__(*args, **kwargs, infinite_loader=infinite_loader,
                         location=location, text_column_name="caption")
