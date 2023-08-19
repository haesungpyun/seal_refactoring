import random
from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.training.callbacks import TrainerCallback
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
import os
import logging
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from allennlp.data import TensorDict


logger = logging.getLogger(__name__)

@TrainerCallback.register("infer_unlabeled")
class InferUnlabeledCallback(TrainerCallback):
    
    def __init__(self, serialization_dir: str) -> None:
        self.serialization_dir = serialization_dir
        self.trainer: Optional['GradientDescentTrainer'] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def on_start(self, trainer: GradientDescentTrainer, is_primary: bool = True, **kwargs) -> None:
        ...
        
    def on_batch(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: float or None = None,
        **kwargs
    ) -> None:

        if is_training:
            store_dict = defaultdict(list)
            
            dir_path = os.path.join(self.serialization_dir, './scores/')
            if (not os.path.isfile(dir_path)) and (not os.path.exists(dir_path)):
                os.makedirs(dir_path)     
            
            for output in batch_outputs:
                if output.get("score") is None or output.get("prob") is None:
                    continue
                store_dict["prob"].extend(output.get("prob"))
                store_dict["score"].extend(output.get("score"))
            
            store_dict['prob'] = torch.stack(store_dict['prob'])
            store_dict["score"] = torch.stack(store_dict["score"])
                
            for key in store_dict.keys():
                file_path = os.path.join(dir_path, f'./epoch_{epoch}_training_{key}.txt')
                with open(file_path, 'ab') as f:
                    f.write(b"\n")
                    np.savetxt(f, store_dict[key].cpu().numpy())
                                
    def on_epoch(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs
    ) -> Any:
        
        storage = ThresholdingCallback.storage
        storage['epoch'] = epoch
        self.trainer = trainer        
        threshold_dict = trainer.model.thresholding.as_dict()
        use_scaling = self.trainer.model.loss_fn.loss_scaling["use_scaling"]
        scaler_type = self.trainer.model.loss_fn.loss_scaling["scaler_type"]
        
        if threshold_dict.get('method') != "score":
            logger.info("!!!! No Scores to be Calculated !!!!")
            self.write_reset_score()
            return False
        
        score_name = threshold_dict['score_conf']['score_name']
        cut_type = threshold_dict['score_conf']['cut_type']
        
        storage['threshold'] = threshold_dict['score_conf']['threshold']
        
        if cut_type == 'discrim' :
            logger.info("Calculate Threshold from pos, neg dist using Logistic Regression")
           
            pos_list = storage[f'{score_name}_pos']
            neg_list = storage[f'{score_name}_neg']
            
            if len(pos_list) == 0 or len(neg_list) == 0:
                self.write_reset_score()
                return False
            
            pos, neg = self.sample_list_to_tensor(pos_list, neg_list, num_sample=None)
                            
            x = torch.cat((pos, neg))
            y = torch.cat((torch.zeros(pos.shape[0]), torch.ones(neg.shape[0])))    # 0 -> pos // 1 -> neg
        
            classifier = LogisticRegression()
            classifier.fit(x.cpu(),y.cpu())
            
            b, w = classifier.intercept_, classifier.coef_
            decision_boundary = torch.from_numpy(-b/w).to(self.device)
            threshold_dict['score_conf']['threshold'] = decision_boundary
            
            logger.info(f"Threshold(decision boundary): {decision_boundary}")
                   
            if use_scaling:
                prob_pos_list = storage[f'prob_pos']
                prob_pos = self.list_to_tensor(prob_pos_list)
                if scaler_type == "robust":
                    params = [prob_pos.quantile(0.5), prob_pos.quantile(0.75), prob_pos.quantile(0.25)]
                elif scaler_type == "standard":
                    params = [prob_pos.mean(), prob_pos.std(), 0]
                elif scaler_type == "min_max":
                    params = [prob_pos.max(), ]
                self.trainer.model.loss_fn.loss_scaling["params"] = params
                
            # Erase scores from previous epochs for refinement
            self.write_reset_score()
            storage['threshold'] = decision_boundary
            
            return True

        else:
            logger.info("Calculate Threshold from Scores using Quantile")
            
            pos = torch.stack(storage[f'{score_name}_pos']).to(self.device)
                       
            if pos.numel() == 0:
                self.write_reset_score()
                return False
            
            threshold = pos.quantile(threshold_dict['score_conf']['quantile'])
            threshold_dict['score_conf']['threshold'] = threshold
            
            logger.info(f"Threshold(quantile={threshold_dict['score_conf']['quantile']}): {threshold}")
            
            if use_scaling:
                prob_pos_list = storage[f'prob_pos']
                prob_pos = self.list_to_tensor(prob_pos_list)
                if scaler_type == "robust":
                    params = [prob_pos.quantile(0.5), prob_pos.quantile(0.75), prob_pos.quantile(0.25)]
                elif scaler_type == "standard":
                    params = [prob_pos.mean(), prob_pos.std(), 0]
                elif scaler_type == "min_max":
                    params = [prob_pos.max(), ]
                elif scaler_type == "mad":
                    params = [prob_pos.quantile(0.5), torch.abs(prob_pos-prob_pos.quantile(0.5)).quantile(0.5), 0]
                self.trainer.model.loss_fn.loss_scaling["params"] = params
            
            # Erase scores from previous epochs for refinement
            self.write_reset_score()
            storage['threshold'] = threshold
            
            return True
    
    def write_reset_score(
        self,
        is_training=False,
        reset = True,
        **kwargs:Any         
    ):  
        """
        cls.storage = {
            "epoch": int,
            "cut_type": str,   # ["all", "quantile", "discrim"]
            "score_name":str,   # ["score", "prob"]
            "score": List[torch.Tensor],
            "prob": List[torch.Tensor],
            "score_pos": List[torch.Tensor],
            "score_neg": List[torch.Tensor],
            "prob_pos": List[torch.Tensor],
            "prob_neg": List[torch.Tensor]
        }
        """
        storage = ThresholdingCallback.storage

        dir_path = os.path.join(self.serialization_dir, './scores/')
        if (not os.path.isfile(dir_path)) and (not os.path.exists(dir_path)):
            os.makedirs(dir_path)           
                            
        ep = storage['epoch']
        for key in storage:
            value = storage[key]
            if type(value) == list:
                file_path = os.path.join(dir_path, f"epoch_{ep}_valid_{key}.pt")
                value = self.list_to_tensor(value)
                torch.save(value, file_path)
                if reset: 
                    storage[key] = []
            else:
                file_path = os.path.join(dir_path, f"epoch_{ep}_valid_{key}.pt")
                torch.save(value, file_path)
                if reset: 
                    storage[key] = []
                
    def list_to_tensor(
        self,
        lists:list
    ):  
        assert type(lists)==list
        if len(lists) == 0:
            return torch.zeros(0, 1).to(self.device)
        elif type(lists[0]) != torch.Tensor:
            return lists
        else:
            return torch.stack(lists).to(self.device)
    
    def sample_list_to_tensor(
        self,
        pos_list,
        neg_list,
        num_sample=None,
    ):
        logger.info("Convert Pos/Neg List to Tensor")      
       
        pos_samples = torch.stack(pos_list).to(self.device)
        neg_samples = torch.stack(neg_list).to(self.device)
        
        # pos_q3, pos_q1 = pos_samples.quantile(0.75), pos_samples.quantile(0.25)
        # pos_iqr = pos_q3 - pos_q1
        
        # neg_q3, neg_q1 = neg_samples.quantile(0.75), neg_samples.quantile(0.25)
        # neg_iqr = neg_q3 - neg_q1
        
        # pos_samples = pos_samples[(pos_q1 - 1.5*pos_iqr <= pos_samples) & (pos_samples <= pos_q3 + 1.5*pos_iqr), None]
        # neg_samples = neg_samples[(neg_q1 - 1.5*neg_iqr <= neg_samples) & (neg_samples <= neg_q3 + 1.5*neg_iqr), None]
        
        if num_sample is None:
            num_sample = min(len(pos_samples), len(neg_samples))
        
        pos_samples = random.sample(list(pos_samples), num_sample)
        neg_samples = random.sample(list(neg_samples), num_sample)
        
        return torch.stack(pos_samples).to(self.device),\
            torch.stack(neg_samples).to(self.device)
            
    def set_reset_state(
        self,
        reset
    ):
        self.reset = reset
        return self.reset    

    @classmethod
    def save_to_storage(
        cls,
        score_conf:Dict,
        word_tags: List[List[str]],
        batch_bio_gold_tags:List[List[str]],
        buffer: Dict,
        is_training=True,
        **kwargs:Any        
    ):  
                    
        assert len(word_tags) == len(batch_bio_gold_tags)
        assert type(word_tags) == type(batch_bio_gold_tags) == list
        
        if score_conf is not None:
            cls.storage['cut_type'] = score_conf.get('cut_type', 'discrim')
            cls.storage['score_name'] = score_conf.get('score_name', "score")
        
        score = buffer.get("score")
        prob = buffer.get('prob')
        
        cls.storage["score"].extend(score)
        cls.storage['prob'].extend(prob)
        
        pos_idx, neg_idx = [], []
        for i, (pred, gold) in enumerate(zip(word_tags, batch_bio_gold_tags)):
            if pred==gold:
                pos_idx.append(i)
            else:
                neg_idx.append(i)
        pos_idx, neg_idx = torch.LongTensor(pos_idx), torch.LongTensor(neg_idx)
        
        score_pos, score_neg = score[pos_idx,], score[neg_idx,]
        prob_pos, prob_neg = prob[pos_idx,], prob[neg_idx,]
        
        cls.storage['score_pos'].extend(score_pos)
        cls.storage['score_neg'].extend(score_neg)
        cls.storage['prob_pos'].extend(prob_pos)
        cls.storage['prob_neg'].extend(prob_neg)
        cls.storage['pos_words'].extend([buffer['meta'][i]['words'] for i in pos_idx])
        cls.storage['neg_words'].extend([buffer['meta'][i]['words'] for i in neg_idx])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "threshold": ThresholdingCallback(self.serialization_dir).storage['threshold']
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ThresholdingCallback(self.serialization_dir).storage['threshold'] = state_dict['threshold']
 