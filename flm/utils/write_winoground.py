import json
import pandas as pd
import pyarrow as pa
import os

from tqdm import tqdm
from collections import defaultdict


def process(root, iden, row):
    text0 = row[0]["caption_0"]
    text1 = row[0]["caption_1"]
    img0_name = row[0]["image_0"]
    img1_name = row[0]["image_1"]
    img0_path = f"{root}/data/images/{img0_name}.png"
    img1_path = f"{root}/data/images/{img1_name}.png"
    # collapsed_tag = row[0]["collapsed_tag"]
    with open(img0_path, "rb") as fp:
        img0 = fp.read()
    with open(img1_path, "rb") as fp:
        img1 = fp.read()

    # texts = [r["sentence"] for r in row]
    # labels = [r["label"] for r in row]

    # split = iden.split("-")[0]

    # if iden.startswith("train"):
    #     directory = row[0]["directory"]
    #     path = f"{root}/images/train/{directory}/{iden}"
    # else:
    #     path = f"{root}/{split}/{iden}"

    # with open(f"{path}-img0.png", "rb") as fp:
    #     img0 = fp.read()
    # with open(f"{path}-img1.png", "rb") as fp:
    #     img1 = fp.read()

    return [img0, img1, text0, text1, iden]


def make_arrow(root, dataset_root):
    # train_data = list(
    #     map(json.loads, open(f"{root}/data/examples.jsonl").readlines())
    # )
    test1_data = list(
        map(json.loads, open(f"{root}/data/examples.jsonl").readlines())
    )
    # dev_data = list(map(json.loads, open(f"{root}/nlvr2/data/dev.json").readlines()))

    # balanced_test1_data = list(
    #     map(
    #         json.loads,
    #         open(f"{root}/nlvr2/data/balanced/balanced_test1.json").readlines(),
    #     )
    # )
    # balanced_dev_data = list(
    #     map(
    #         json.loads,
    #         open(f"{root}/nlvr2/data/balanced/balanced_dev.json").readlines(),
    #     )
    # )

    # unbalanced_test1_data = list(
    #     map(
    #         json.loads,
    #         open(f"{root}/nlvr2/data/unbalanced/unbalanced_test1.json").readlines(),
    #     )
    # )
    # unbalanced_dev_data = list(
    #     map(
    #         json.loads,
    #         open(f"{root}/nlvr2/data/unbalanced/unbalanced_dev.json").readlines(),
    #     )
    # )
    splits = ['test']
    datas = [test1_data]

    # splits = [
    #     "train",
    #     "dev",
    #     "test1",
    #     "balanced_dev",
    #     "balanced_test1",
    #     "unbalanced_dev",
    #     "unbalanced_test1",
    # ]

    # datas = [
    #     train_data,
    #     dev_data,
    #     test1_data,
    #     balanced_dev_data,
    #     balanced_test1_data,
    #     unbalanced_dev_data,
    #     unbalanced_test1_data,
    # ]

    annotations = dict()

    for split, data in zip(splits, datas):
        _annot = defaultdict(list)
        for row in tqdm(data):
            _annot[row["id"]].append(row)
        annotations[split] = _annot

    for split in splits:
        bs = [
            process(root, iden, row) for iden, row in tqdm(annotations[split].items())
        ]

        dataframe = pd.DataFrame(
            bs, columns=["image_0", "image_1", "text0", "text1", "identifier"],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/winoground_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


make_arrow('/group/30042/wybertwang/dataset/winoground',
           '/group/30042/wybertwang/dataset/METER_task_arrow')
