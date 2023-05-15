import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob


def path2rest(path, iid2captions, data_dir, split):
    # split, _, name = path.split("/")[-3:]
    # split = split.split("_")[-1]
    # iid = name
    iid = path

    with open(_get_video_path(path, data_dir, split)[0], "rb") as fp:
        binary = fp.read()

    captions = iid2captions[iid]

    return [
        binary,
        captions,
        iid,
        split,
    ]


def _get_caption(sample):
    return sample[0]


def _get_video_path(file_name, data_dir, split):
    # conceptual captions uses this hashing to create the filename
    rel_dir = 'training'
    if split != 'train':
        rel_dir = 'validation'
    rel_fp = os.path.join(rel_dir, file_name)
    return os.path.join(data_dir, rel_fp), rel_fp


def make_arrow(dataset_root, save_folder, split='train'):
    metadata_dir = os.path.join(dataset_root, 'metadata')
    split_files = {
        'train': 'cc3m_training_success_full.tsv',
        'val': 'cc3m_validation_success_full.tsv',            # there is no test
    }
    split_folders = {'train': 'training',
                     'val': 'validation',  # there is no tes
                     }

    # for split in ["val", "train"]:
    if True:
        target_split_fp = split_files[split]
        metadata = pd.read_csv(os.path.join(
            metadata_dir, target_split_fp), sep='\t')

        # meta_data_path = f"{root}/metadata/cc3m_{split_files}_success_full.tsv"
        # metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')

        # with open(, "r") as fp:
        #     captions = json.load(fp)

        # iid2captions = dict()
        # for cap in tqdm(captions):
        #     iid = cap[0].split("/")[-1]
        #     iid2captions[iid] = [cap[1]]

        iid2captions = dict()
        for item in range(metadata.shape[0]):
            sample = metadata.iloc[item]
            caption = _get_caption(sample)
            iid = sample[1]
            iid2captions[iid] = caption

        # paths = list(glob(f"{dataset_root}/{split_folders[split]}/*"))
        # random.shuffle(paths)

        # caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
        caption_paths = list(iid2captions.keys())
        random.shuffle(caption_paths)

        # if len(paths) == len(caption_paths):
        #     print("all images have caption annotations")
        # else:
        #     print("not all images have caption annotations")
        # print(
        #     len(paths), len(caption_paths), len(iid2captions),
        # )

        sub_len = int(len(caption_paths) // 100000)
        subs = list(range(sub_len + 1))
        for sub in subs:
            sub_paths = caption_paths[sub * 100000: (sub + 1) * 100000]
            bs = [path2rest(path, iid2captions, dataset_root, split)
                  for path in tqdm(sub_paths)]

            dataframe = pd.DataFrame(
                bs, columns=["image", "caption", "image_id", "split"],
            )

            table = pa.Table.from_pandas(dataframe)

            os.makedirs(save_folder, exist_ok=True)
            with pa.OSFile(
                f"{save_folder}/conceptual_caption_{split}_{sub}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del bs
            gc.collect()
