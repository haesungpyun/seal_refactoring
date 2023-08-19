import torch
# import allennlp.data
# import importlib
# from torch.utils.data import Dataset
# import pkgutil
# from torch import nn
# nn.Linear(2,3)
# import importlib.util
# import sys

# # oracle_fn = importlib.__import__('seal.modules.oracle_value_function.oracle_value_function.py')
# # print(oracle_fn)

# seal = importlib.import_module('seal')
# path = getattr(seal, "__path__", [])
# for _, name, ispkg in pkgutil.walk_packages(path):
#     importlib.import_module('seal'+'.'+name)

# spec = importlib.util.spec_from_file_location("seal.dataset", "/home/jylab_intern001/seal_refactoring/seal/custom_dataset.py")
# foo = importlib.util.module_from_spec(spec)
# print(type(foo))
# sys.modules["seal.dataset"] = foo
# spec.loader.exec_module(foo)
# print(type(foo))
# print(getattr(foo, "MNISTDataset"))
# # print(registerd_datareaders)


class MyScoreNN(torch.nn.Module):
    def __init__(
        self,
        encoder : torch.nn.Module,
        projection: torch.nn.Module,
        loss_fn: torch.nn.Module
    ):
        super.__init__()
        
        self.encoder = encoder
        self.proj = projection
        self.loss = loss_fn

    def forward(self, x, y):
        x = self.encoder(x)
        y_pred = self.proj(x, y)
        loss = self.loss(y_pred, y)
        return y_pred, loss

scorenn = MyScoreNN
print(scorenn)
args = getattr(scorenn, "__args__", [])
print(args)
