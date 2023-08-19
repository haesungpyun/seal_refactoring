# import torch, random
# from seal.wrappers.wrapper import Wrapper
# from .exp_architecture import MLP, train, test
# from .exp_seal import MyScoreNN, MyTaskNN
# from torch import nn, optim
# import numpy as np
# from tqdm import tqdm

# class Exp_toy(object):
#     def __init__(
#         self,
#         config_dict
#     ):
#         self.config_dict = config_dict
#         self.wrapper = Wrapper(config_dict=self.config_dict)

#     def make_dataloader(self):
#         data_reader, constructor, args = self.wrapper.import_module('dataset_reader')
#         data_loader = self.wrapper.import_module(class_name='data_loader', dataset=data_reader)
#         return data_loader
    
#     def set_experiment(self):
        
#         model, constructor, args = self.wrapper.import_module('dataset_reader')
#         args = self.wrapper.check_module_args(model, constructor, args)
#         model = self.wrapper.construct_module(model, constructor, args)


# # get training hyper parameter 
# def get_training_params(self, config_dict):
#     self.config_dict = config_dict
#     return self.confing_dict

# training_params = get_training_params(config_dict)
# trainset, testset = random.split(datasets, [training_params['train_size'], 
#                                     training_params['test_size']],
#                            generator = torch.Generator().manual_seed(42))
# train_dataloader = dataloader(trainset, batch_size = training_params['batch'], shuffle = True)
# test_dataloader = dataloader(testset, batch_size = training_params['batch'], shuffle = True)

# loss_fn = nn.CrossEntropyLoss()
# feature_net = MLP(dataloader=dataloader)
# tasknn = MyTaskNN(model)
# scorenn  = MyScoreNN(model, loss_fn)


# # Set the Training Parameters
# optimizer = getattr(optim, training_params['optimizer'])(model.parameters(), 
#                                                          lr = training_params['lr'])

# # Train the Network
# for t in range(training_params['epoch']):
#     print(f'----- Epoch {t+1} -----')
#     train(train_dataloader, tasknn, scorenn, optimizer)
#     accuracy, loss = test(test_dataloader,tasknn, scorenn)





