import codecs
import re

import torch
from .constructor import Constructor


class DataConstructor(Constructor):
    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        
    def make_dataloader(self):
        dataset = self.construct_class(self.config_dict['dataset_reader'], 
                                       'dataset_reader')

        if "allennlp" in str(self.config_dict['data_loader']):
            data_loader = self.construct_class(self.config_dict['data_loader'], 
                                               'data_loader', 
                                               reader=dataset)
        else:
            data_loader = self.construct_class(self.config_dict['data_loader'], 
                                               'data_loader', 
                                               dataset=dataset)
        
        if "allennlp" in str(self.config_dict['data_loader']):
            instance_generator = (
                instance
                for key, data_loader in data_loader.items()
                for instance in data_loader.iter_instances()
            )
            vocab = self.construct_class(self.config_dict['vocabulary'],"vocabulary", instances=instance_generator)
            data_loader.index_with(vocab)
                     
        # 패키지 별로 data loading format 동일하게 후처리 wrapping 해주기

        return data_loader, vocab


