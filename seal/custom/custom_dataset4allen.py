import glob
import pickle
import torch
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ArrayField

from typing import List

class MNISTDataset4Allen(DatasetReader):
    def __init__(
        self,
        path_to_data:str,
        test:bool = False,
        **kwargs
    ):
        super().__init__()
        self.path_to_data = path_to_data
        self.test = test
        if test:
            self.num_tags = 10
        else:
            self.num_tags=19

    def read(self,datapath):
        with open(datapath, 'rb') as f:
            dataset = pickle.load(f)
    
        self.nums = dataset[0]
        self.labels = torch.nn.functional.one_hot(
            dataset[1],
            self.num_tags
        )

        for num, label in zip(self.nums, self.labels):
            assert num.shape == (2, 28, 28)

            yield Instance({
                'number': ArrayField(num),
                'labels': ArrayField(label.float())
            })
    
    


"""
train_dataset = MNISTDataset(train_data, test=False)
val_dataset = MNISTDataset(valid_data, test=False)
test_dataset = MNISTDataset(test_data, test=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
"""    