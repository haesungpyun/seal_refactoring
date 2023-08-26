import codecs
import re

import torch
from .constructor import Constructor


class DataConstructor(Constructor):
    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        
    def make_dataloader(self):
        dataset = self.construct_class(self.config_dict['dataset_reader'], 'dataset_reader')
        data_loader = self.construct_class(self.config_dict['data_loader'], 'data_loader', reader=dataset)
        
        if "allennlp" in str(data_loader.__module__):
            instance_generator = (
                instance
                for key, data_loader in data_loader.items()
                for instance in data_loader.iter_instances()
            )
            vocab = self.construct_class(self.config_dict['vocabulary'],"vocabulary")
            data_loader.index_with(vocab)
        else:
            self.label_dictionary = {}
            _NEW_LINE_REGEX = re.compile(r"\n|\r\n")
            if self.config_dict['vocabulary'].get('filename', False):
                filename = self.config_dict['vocabulary'].get('filename')
                with codecs.open(filename, "r", "utf-8") as input_file:
                    lines = _NEW_LINE_REGEX.split(input_file.read())
                    # Be flexible about having final newline or not
                    if lines and lines[-1] == "":
                        lines = lines[:-1]
                    for index, line in enumerate(lines):
                        token = line.replace("@@NEWLINE@@", "\n")
                        self.label_dictionary[token] = index
            else:
                for index, batch in data_loader:
                    label = batch['label']
                    if self.label_dictionary.get(label, None):
                        continue
                    else:
                        self.label_dictionary[label] = index
                     
            tokenizer = self.construct_class(self.config_dict['tokenizer', "tokenizer"])
            for i, batch in enumerate(data_loader):
                # 직접 batch tensor를 조작하는 게 맞나...?
                # dataset/dataloader 단에서 iter 하는 instance들에 적용하는 게 깔끔할 거 같은데
                batch['x'] = tokenizer(batch['x'], return_tensor='pt')
                batch['label'] = torch.tensor([self.label_dictionary[i] for i in batch['label']])
                
        # 패키지 별로 data loading format 동일하게 후처리 wrapping 해주기

        return data_loader, vocab


