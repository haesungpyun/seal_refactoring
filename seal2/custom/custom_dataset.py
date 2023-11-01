from collections import defaultdict
import os
import codecs
import numpy as np
from wcmatch import glob
import json
import pathlib
import re
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer
from typing import Any, List

_NEW_LINE_REGEX = re.compile(r"\n|\r\n") 

class MyDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        return_tensors: str = "pt",
        truncate: bool = True,
        max_length: int = 512,
        padding: bool = False,
        label_dict_path: str = None,
        tr_vl_ts:str = 'train',
        label_txt_path:str = None,
        data_path:str = './',
        dataset: torch.utils.data.Dataset = None,
        random_shuffle: bool = False,
        **kwargs
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = {
            "return_tensors": return_tensors,
            "truncation": truncate,
            "max_length": max_length,
            "padding": padding
        }
        self.label_dict_path = label_dict_path

        self.tr_vl_ts = tr_vl_ts

        self.data_path = pathlib.Path(data_path)        
        self.dataset = dataset
        self.data_len = 0

        # Load label dictionary if it exists
        self.label_txt_path = label_txt_path
        self._label_dictionary = defaultdict(int)
        
        self.random_shuffle = random_shuffle

        self._load_label_dict()

    def __iter__(self):
        if not self.dataset:
            for file_ in glob.glob(self.data_path.absolute().as_posix(), flags=glob.EXTGLOB):
                with codecs.open(file_, "r", "utf-8") as f:
                    lines = _NEW_LINE_REGEX.split(f.read())
                    for idx in np.random.permutation(len(lines)) if self.random_shuffle else range(len(lines)):
                        example = json.loads(lines[idx])
                        instance = self._read_data(**example)
                        yield instance
            self._save_label_dict()
            
        else:
            for idx in np.random.permutation(len(lines)) if self.random_shuffle else range(self.dataset[self.tr_vl_ts]['text']):
                example = {
                    "title": self.dataset[self.tr_vl_ts]['text'][idx],
                    "labels":self.dataset[self.tr_vl_ts]['label'][idx]
                }
                instance = self._read_data(**example)
                yield instance
            self._save_label_dict()
            
    def _read_data(
        self,
        title: str,
        body: str,
        topics: str,
        idx: str,
        **kwargs: Any,
    ): 
        labels = []
        for value in topics.values():
            labels += (
                [value]  # type:ignore
                if not isinstance(value, list)
                else value
            )
        
        label_idx_list = [] 
        for label in labels:
            if self._label_dictionary.get(label) is None:
                self._label_dictionary[label] = len(self._label_dictionary)
            label_idx_list.append(self._label_dictionary[label])

        return {'text': body, 'labels':label_idx_list}

    def _load_label_dict(self):
        if os.path.exists(os.path.join(self.data_path.parent,'./label_dict.txt')):
            with open(os.path.join(self.data_path.parent,'./label_dict.txt'), 'r') as f:
                self._label_dictionary = json.load(f)
            return
                    
        if self.label_txt_path:    
            with codecs.open(self.label_txt_path , "r", "utf-8") as f:
                lines = _NEW_LINE_REGEX.split(f.read())
                # Be flexible about having final newline or not
                if lines and lines[-1] == "":
                    lines = lines[:-1]
                for index, line in enumerate(lines):
                    token = line.replace("@@NEWLINE@@", "\n")
                    self._label_dictionary[token] = index
            return
        raise ValueError('label_txt_path is None')

    def _save_label_dict(self):
        with open(os.path.join(self.data_path.parent,'./label_dict.txt'), 'w') as f:
            json.dump(self._label_dictionary, f)
