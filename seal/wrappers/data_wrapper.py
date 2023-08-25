import importlib
import pkgutil
import sys

import torchvision
from allennlp.data import allennlp_collate
from datasets import load_dataset
from .wrapper import Wrapper


class DataWrapper(Wrapper):
    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        
    def make_dataloader(self):
        dataset = self.construct_class(self.config_dict['dataset_reader'], 'dataset_reader')
        data_loader = self.construct_class(self.config_dict['data_loader'], 'data_loader', reader=dataset)

        # if allennlp:
        #     vocabulary_ = vocabulary.construct(instances=instance_generator)
        #     data_loader.index_with(vocabulary_)
        # else:
            # tokenizer = self.construct_module(self.config_dict['tokenizer'])

        #     for (x, y, z) in data_loader:
                # x = tokenizer.tokenize(x)
                # y = self._indexed_labels = [
                #     vocab.get_token_index(label, self._label_namespace)  # type: ignore
                #     for label in self.labels
                # ]

                # z = z

        return data_loader, 


