from ast import Dict
import codecs
import glob
import json
import pathlib
import pickle
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.nn import functional as F

class MyDataset(Dataset):
    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        train:bool = True,
        label_txt_path:str = None,
        data_path:str = './',
        dataset = None,
        **kwargs
    ):
        super().__init__()
        self.data_path = pathlib.Path(data_path)
        self.tokenizer = tokenizer
        self.train = train
        self.label_txt_path = label_txt_path
        self.dataset = dataset

        self.label_set = set()

        self.read_data()
        self.make_dataset()

    def read_data(self):
        if not self.dataset:
            for file_ in glob.glob(self.data_path, flags=glob.EXTGLOB):
                # logger.info(f"Reading {file_}")
                with open(file_, encoding="utf-8") as f:
                    for line in f:
                        example = json.loads(line)
                        instance = self.tokenize_data(**example)
                        yield instance
        else:
            if self.train:
                for i in range(self.dataset['train']['text']):
                    example = {
                        "title": self.dataset['train']['text'][i],
                        "labels":self.dataset['train']['label'][i]
                    }
                    instance = self.tokenize_data(**example)
                    yield instance
            else:
                for i in range(self.dataset['test']['text']):
                    example = {
                        "title": self.dataset['test']['text'][i],
                        "labels":self.dataset['test']['label'][i]
                    }
                    instance = self.tokenize_data(**example)
                    yield instance

    def tokenize_data(
        self,
        title: str,
        body: str,
        topics: str,
        idx: str,
    ):
        tokens = self.tokenizer(title, return_tensors='pt', truncation=True, max_length=512)
        labels_ = []
        for value in topics.values():
            labels_ += (
                [value]  # type:ignore
                if not isinstance(value, list)
                else value
            )
        self.label_set.update(labels_)

        return {'x': tokens, 'labels':labels_}

    def make_dataset(self):
        instance_gen_ = self.read_data()
        
        self.label_dictionary = {}
        if self.train:
            if self.label_txt_path:        
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
            else:
                label_list = list(self.label_set)
                self.label_dictionary = {label: idx for idx, label in enumerate(label_list)}

            with open(self.data_path.parent + './label_dict.txt', 'w') as f:
                json.dump(self.label_dictionary)
   
        self.total_data = []
        for tokens, labels_ in instance_gen_:
            label_idx = []
            for label in labels_:
                label_idx.append(self.label_dict[label])
            self.total_data.append({'x':tokens, 'labels':label_idx})
    
    def __len__(self,):
        return len(self.total_data)
    
    def __getitem__(self, idx):
        return self.total_data[idx]['x'],\
              self.total_data[idx]['labels']


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
