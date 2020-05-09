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
