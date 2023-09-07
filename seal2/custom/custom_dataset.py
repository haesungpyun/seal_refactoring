from collections import defaultdict
import os
import codecs
from wcmatch import glob
import json
import pathlib
import pickle
import re
import torch
from torch.utils.data import Dataset
from typing import Any, List
from transformers import PreTrainedTokenizer


class MyDataset(Dataset):
    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        tr_vl_ts:str = 'train',
        label_txt_path:str = None,
        data_path:str = './',
        dataset = None,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tr_vl_ts = tr_vl_ts
        self.label_txt_path = label_txt_path
        
        self.data_path = pathlib.Path(data_path)        
        self.dataset = dataset

        self.data_generator = self.make_dataset()

    def _read_data(self):
        if not self.dataset:
            # for file_ in os.listdir(self.data_path):
                # logger.info(f"Reading {file_}")
            # !!! should deal with multiple files in the future !!!
            with open(self.data_path,'r', encoding="utf-8") as f:
                for line in f:
                    example = json.loads(line)
                    instance = self._tokenize_data(**example)
                    yield instance
        else:
            for i in range(self.dataset[self.tr_vl_ts]['text']):
                example = {
                    "title": self.dataset[self.tr_vl_ts]['text'][i],
                    "labels":self.dataset[self.tr_vl_ts]['label'][i]
                }
                instance = self._tokenize_data(**example)
                yield instance
            
    def _tokenize_data(
        self,
        title: str,
        body: str,
        topics: str,
        idx: str,
        **kwargs: Any,
    ):
        tokens = self.tokenizer(title, return_tensors='pt', truncation=True, max_length=512)
        labels_ = []
        for value in topics.values():
            labels_ += (
                [value]  # type:ignore
                if not isinstance(value, list)
                else value
            )

        return {'x': tokens, 'labels':labels_}

    def _load_label_dict(self):
        self.label_dictionary = defaultdict(int)
        _NEW_LINE_REGEX = re.compile(r"\n|\r\n") 
        filename = self.label_txt_path
        with codecs.open(filename, "r", "utf-8") as f:
            lines = _NEW_LINE_REGEX.split(f.read())
            # Be flexible about having final newline or not
            if lines and lines[-1] == "":
                lines = lines[:-1]
            for index, line in enumerate(lines):
                token = line.replace("@@NEWLINE@@", "\n")
                self.label_dictionary[token] = index

    def _save_label_dict(self):
        with open(self.data_path.parent + './label_dict.txt', 'w') as f:
                json.dump(self.label_dictionary)
        
    def make_dataset(self):        
        instance_gen_ = self._read_data()
        
        self._load_label_dict()

        for tokens, labels_ in instance_gen_:
            label_idx_list = []
            for label in labels_:
                if self.tr_vl_ts != 'test' and self.label_dictionary.get(label) is None:
                    self.label_dictionary[label] = len(self.label_dictionary)
                label_idx_list.append(self.label_dictionary[label])
            yield {'x':tokens, 'labels':label_idx_list}

        self._save_label_dict()
    
    def __len__(self,):
        return len(self.data_generator)
    
    def __getitem__(self, idx):
        isinstance = next(self.data_generator)
        return isinstance['x'], isinstance['labels']

class MNISTDataset(Dataset):
    def __init__(
        self,
        path_to_data:str,
        test:bool = False,
        dataset:Dataset=None,
        **kwargs
    ):
        super().__init__()
        with open(path_to_data, 'rb') as f:
            self.dataset = pickle.load(f)
        
        self.num_pairs = self.dataset[0]
        self.labels = torch.nn.functional.one_hot(
            self.dataset[1],
            19
        )
        
    def __len__(self,):
        return len(self.num_pairs)
    
    def __getitem__(self, idx):
        return self.num_pairs[idx].float(), self.labels[idx].float()
