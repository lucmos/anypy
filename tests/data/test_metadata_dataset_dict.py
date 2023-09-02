from pathlib import Path

import torch
from datasets import Dataset

from anypy.data.metadata_dataset_dict import MetadataDatasetDict


def test_dataset_dict(tmp_path: Path):
    train_dataset = Dataset.from_dict({"x": torch.randn(100, 50), "label": torch.randint(50, (100,))})
    test_dataset = Dataset.from_dict({"x": torch.randn(100, 50), "label": torch.randint(50, (100,))})

    my_dataset = MetadataDatasetDict(train=train_dataset, test=test_dataset)

    assert my_dataset.keys() == ["train", "test"]

    tmp_path.mkdir(exist_ok=True)

    my_dataset["metadata"] = {"num_classes": 100, "model": None}

    my_dataset.save_to_disk(tmp_path)

    assert my_dataset.keys() == ["train", "test"]

    my_dataset = MetadataDatasetDict.load_from_disk(tmp_path)

    assert my_dataset.keys() == ["train", "test"]

    my_dataset = my_dataset.map(lambda x: {"new_column": x["label"] * 2})

    assert my_dataset["train"]["new_column"][0] == my_dataset["train"]["label"][0] * 2
    assert my_dataset["test"]["new_column"][0] == my_dataset["test"]["label"][0] * 2

    my_dataset = my_dataset.filter(lambda x: x["label"] < 10)

    my_dataset.set_format("numpy", columns=["label"])
    assert (my_dataset["train"]["label"] < 10).all()
    assert (my_dataset["test"]["label"] < 10).all()
