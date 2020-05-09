import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import torch
import tqdm
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, torch_distributed_zero_first
logger = logging.getLogger(__name__)

@dataclass
class ThredditInputExample(object):
    """A single training/test example for the Threddit dataset."""
    """We replicate the parent four times so that the model sees how each parent is related to the ith child comment"""

    def __init__(self, example_id, response, parent, child_comment_0, child_comment_1, child_comment_2, child_comment_3, label=None):
        self.example_id = example_id
        self.response = response
        self.parent = [parent, parent, parent, parent]
        self.children = [
            child_comment_0,
            child_comment_1,
            child_comment_2,
            child_comment_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        attributes = [
            "id: {}".format(self.example_id),
            "response: {}".format(self.response),
            "parent: {}".format(self.parent),
            "child_0": {}".format(self.children[0]),
            "child_1": {}".format(self.children[1]),
            "child_2": {}".format(self.children[2]),
            "child_3": {}".format(self.children[3]),
        ]

        if self.label is not None:
            attributes.append("label: {}".format(self.label))

        return ", ".join(attributes)

@dataclass(frozen=True)
class ThredditInputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    
    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]

class ThredditProcessor(DataProcessor):
    """Processor for the Reddit analysis project."""

    def get_example_from_tensor_dict(self, tensor_dict):
        example_id = tensor_dict["idx"].numpy()
        response = tensor_dict["response"].numpy()
        parent = tensor_dict["parent"].numpy().decode("utf-8")
        label = str(tensor_dict["label"].numpy())

        return ThredditInputExample(
            example_id,
            response,
            parent,
            tendor_dict["child_comment_0"].numpy().decode("utf-8"),
            tensor_dict["child_comment_1"].numpy().decode("utf-8"),
            tensor_dict["child_comment_2"].numpy().decode("utf-8"),
            tensor_dict["child_comment_3"].numpy().decode("utf-8"),
            label
        )

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl"))) 
        with open(os.path.join(data_dir, "train.jsonl"), "r") as f:
            return self._create_examples(f.read().splitlines(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "val.jsonl"), "r") as f:
            return self._create_examples(f.read().splitlines(), "val")

    def get_labels(self):
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            line = loads(line)
            guid = "%s-%s" % (set_type, i)
            try:
                response = line["response"]
                parent = line["parent"]
                cc0 = line["child_comment_0"]
                cc1 = line["child_comment_1"]
                cc2 = line["child_comment_2"]
                cc3 = line["child_comment_3"]
                label = line["label"]
            except IndexError:
                continue
            examples.append(
                ThredditInputExample(
                    guid,
                    response,
                    parent,
                    cc0,
                    cc1,
                    cc2,
                    cc3,
                    label
                )
            )
        return examples