import torch
import importlib
from seal.wrappers.wrapper import Wrapper
from seal.wrappers.model_wrapper import ModelWrapper
from seal.wrappers.data_wrapper import DataWrapper
from seal.wrappers.wrapper import Wrapper
from allennlp.data import Vocabulary

class Experiment(object):
    def __init__(
            self,
            config_dict
    ):
        self.config_dict = config_dict
        self.wrapper = Wrapper()
        self.data_wrapper = DataWrapper(config_dict=self.config_dict)
        self.model_wrapper = ModelWrapper(config_dict=self.config_dict)

    def set_experiment(self):

        data_loader = self.data_wrapper.make_dataloader()

        vocab = self.wrapper.construct_class(self.config_dict['vocabulary'],"vocabulary")

        seal = self.model_wrapper.make_shared_nn_seal(vocab)

        optimizers = self.make_optimizers(seal, self.config_dict['trainer'].pop('optimizers'))

        return data_loader, seal, optimizers
    
    def make_optimizers(self, model, optimizers_config):
        
        optimizers = {
            name:
            self.wrapper.construct_class(
                optimizers_config[name], 
                name, 
                params=getattr(
                    getattr(
                        model, 
                        'inference_module'
                    ),
                    name
                ).parameters()
            )
            for name in optimizers_config.keys()
        }
        return optimizers