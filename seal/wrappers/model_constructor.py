import difflib
from typing_extensions import override
from .constructor import Constructor


class ModelConstructor(Constructor): # type: ignore
    def __init__(self, config_dict):
        super().__init__()
        self.model_config = config_dict['model']
    
    def make_shared_nn_seal(self, vocab):
        task_net = self.construct_class(self.model_config['args']['task_nn'], 'task_nn')
        score_net = self.construct_class(self.model_config['args']['score_nn'], 'score_nn')
        oracle_value_function = self.construct_class(self.model_config['args']['oracle_value_function'], 'oracle_value_function')
        
        inference_module = self.construct_class(
            self.model_config['args']['inference_module'], 
            'inference_module',
            task_net=task_net,
            score_net=score_net
        )
        
        sampler_ = self.construct_class(
                    score_nn=score_net,
                    oracle_value_function=oracle_value_function,
        )
        sampler_.append(inference_module)
        
        inference_module = self.construct_class(
            self.model_config['args']['inference_module'], 
            'inference_module',
            task_net=task_net,
            score_net=score_net
        )
        
        loss_fn = self.construct_class(
            self.model_config['args']['loss_fn'], 
            'loss_fn',
            score_net=score_net
        )
        
        seal = self.construct_class(
            self.model_config, 
            'model',
            vocab=vocab,
            sampler=sampler_,
            oracle_value_function=oracle_value_function,
            score_nn=score_net,
            inference_module=inference_module,
            loss_fn=loss_fn
        )

        return seal
