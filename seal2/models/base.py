from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Iterator,
    cast,
    Generator,
)
import contextlib
import torch
from seal2.modules.sampler import (
    Sampler,
    SamplerContainer,
    AppendingSamplerContainer,
)
from seal2.common import ModelMode
from seal2.modules.sampler.inference_net import (
    InferenceNetSampler,
)
from seal2.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal2.modules.score_nn import ScoreNN
from seal2.modules.loss import Loss

from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from seal2.modules.logging import (
    LoggingMixin,
    LoggedValue,
    LoggedScalarScalar,
    LoggedScalarScalarSample,
    LoggedNPArrayNPArraySample,
)
from seal2.modules.task_nn import TaskNN
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ScoreBasedLearningModel(LoggingMixin, torch.nn.Module):
    def __init__(
        self,
        sampler: Sampler,
        loss_fn: Loss,
        oracle_value_function: Optional[OracleValueFunction] = None,
        score_nn: Optional[ScoreNN] = None,
        inference_module: Optional[Sampler] = None,
        evaluation_module: Optional[Sampler] = None,
        num_eval_samples: int = 10,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        use_distil: bool = False,
        label_smoothing: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        """

        The model will be used in two ways:

            1. By calling `forward(x, labels)` the inference module is run to produce predictions (and loss if labels are present).
                This will be used to produce predictions as well as to update the parameters of the Sampler.

            2. By calling `update()` the scoreNN loss is computed by first calling sampler in eval model
                (with no_grad) with logging turned off.

            3. By calling `compute_score(x,y)` the score for (x,y) will be computed. This is useful for doing custom evaluations of ScoreNN.
                In order for such evaluations to not interfere with the training of ScoreNN, we need to do these after the optimizer step
                for ScoreNN. Hence, we will use on_batch or on_epoch callback for this.
                All such evaluations should log values in their own attributes,
                and it is their responsibility to add these values to `metrics` so that they can be logged to wandb and console.


        Args:
            loss_fn: Loss function for updating ScoreNN
        """
        super().__init__(
            **kwargs
        )  # type:ignore
        self.sampler = sampler
        self.loss_fn = loss_fn
        self.oracle_value_function = oracle_value_function
        self.score_nn = score_nn

        self._regularizer = regularizer

        if inference_module is not None:
            self.inference_module = inference_module
        else:
            self.inference_module = sampler

        self.evaluation_module = evaluation_module
        self.num_eval_samples = num_eval_samples
        # self.eval_only_metrics = {}

        if initializer is not None:
            initializer(self)
        self.logging_children.append(self.loss_fn)
        self.logging_children.append(self.sampler)
        self.logging_children.append(self.inference_module)

        if evaluation_module is not None:
            self.logging_children.append(self.evaluation_module)

        mode = ModelMode.UPDATE_SCORE_NN

        if self.score_nn is not None:
            for param in self.score_nn.parameters():
                mode.mark_parameter_with_model_mode(param)
        mode = ModelMode.UPDATE_TASK_NN

        if inference_module is not None:
            for param in self.inference_module.parameters_with_model_mode(
                mode
            ):
                mode.mark_parameter_with_model_mode(param)

        for n, p in self.named_parameters():
            if not ModelMode.hasattr_model_mode(p):
                logger.warning(f"{n} does not have ModelMode set.")

        self.use_pseudo_labeling = use_distil
        self.use_ls = label_smoothing.get('use_ls', False)
        self.alpha = label_smoothing.get('alpha', 0.0)
        
    @classmethod
    def from_partial_objects(
        cls,
        sampler: Sampler,
        loss_fn: Loss,
        inference_module: Optional[Sampler] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        evaluation_module: Optional[Sampler] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> "ScoreBasedLearningModel":

        if oracle_value_function is not None:
            sampler_ = sampler.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
            loss_fn_ = loss_fn.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
        else:
            sampler_ = sampler.construct(
                score_nn=score_nn,
            )
            loss_fn_ = loss_fn.construct(
                score_nn=score_nn,
            )

        # if no seperate inference module is given,
        # we will be using the same sampler

        # test-time inference.

        if inference_module is None:
            inference_module_ = sampler_
        else:
            inference_module_ = inference_module.construct(
                score_nn=score_nn,
                oracle_value_function=oracle_value_function,
                main_sampler=sampler_,
            )

        if evaluation_module is not None:
            evaluation_module_ = evaluation_module.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
        else:
            evaluation_module_ = None

        return cls(
            sampler=sampler_,
            loss_fn=loss_fn_,
            oracle_value_function=oracle_value_function,
            score_nn=score_nn,
            inference_module=inference_module_,
            evaluation_module=evaluation_module_,
            regularizer=regularizer,
            initializer=initializer,
            **kwargs,
        )

    def named_parameters_for_model_mode(
        self, mode: ModelMode
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:

        for name, param in self.named_parameters():
            param = cast(torch.nn.Parameter, param)

            if mode.is_parameter_model_mode(param):
                yield (name, param)

    def parameters_for_model_mode(
        self, mode: ModelMode
    ) -> Iterator[torch.nn.Parameter]:
        for param in self.parameters():
            if mode.is_parameter_model_mode(param):
                yield param

    @classmethod
    def from_partial_objects_with_shared_tasknn(
        cls,
        loss_fn: Loss,
        inference_module: Sampler,
        task_nn: TaskNN,
        sampler: Optional[SamplerContainer] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        evaluation_module: Optional[Sampler] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> "ScoreBasedLearningModel":
        """
        This constructor is used only when the `sampler` is an instance of `SamplerContainer`
        and we wish to use tasknn as both the inference_module and a sampler in the constituent sampler.
        """
        infnet_sampler = inference_module.construct(
            inference_nn=task_nn,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
        )

        if oracle_value_function is not None:
            if sampler is None:
                sampler_ = AppendingSamplerContainer(
                    score_nn=score_nn,
                    oracle_value_function=oracle_value_function,
                    constituent_samplers=[],
                    log_key="sampler",
                )
            else:
                sampler_ = sampler.construct(
                    score_nn=score_nn,
                    oracle_value_function=oracle_value_function,
                )
            loss_fn_ = loss_fn.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
        else:
            if sampler is None:
                sampler_ = AppendingSamplerContainer(
                    score_nn=score_nn,
                    constituent_samplers=[],
                    log_key="sampler",
                )
            else:

                sampler_ = sampler.construct(
                    score_nn=score_nn,
                )
            loss_fn_ = loss_fn.construct(
                score_nn=score_nn,
            )
        # add the infnet sampler
        sampler_.append_sampler(infnet_sampler)

        # test-time inference.
        # reconstruct the infnet sampler with shared tasknn weights
        # to set it as the inference_module
        inference_module_ = inference_module.construct(
            inference_nn=task_nn,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
        )
        inference_module_.log_key = inference_module_.log_key + "_inf"

        if evaluation_module is not None:
            evaluation_module_ = evaluation_module.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
        else:
            evaluation_module_ = None

        return cls(
            sampler=sampler_,
            loss_fn=loss_fn_,
            oracle_value_function=oracle_value_function,
            score_nn=score_nn,
            inference_module=inference_module_,
            evaluation_module=evaluation_module_,
            regularizer=regularizer,
            initializer=initializer,
            **kwargs,
        )

    def calculate_metrics(
        self,
        x: Any,
        labels: torch.Tensor,  # shape: (batch, ...)
        y_hat: torch.Tensor,  # shape: (batch, ...)
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:
        return None

    def convert_to_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Converts the labels to one-hot if not already. Required for more complex tasks like sequence tagging."""

        return labels

    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unsqueeze to add a samples dimension"""

        return labels

    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        """Squeeze to remove the samples dimension"""
        raise NotImplementedError

    def initialize_buffer(
        self,
        **kwargs: Any,
    ) -> Dict:
        return {}

    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        kwargs["buffer"] = self.initialize_buffer(**kwargs)

        return kwargs

    def forward(self, **kwargs: Any) -> Dict:
        return self._forward(**self.construct_args_for_forward(**kwargs))

    def _forward(
        self,
        x: Any,
        labels: torch.Tensor,
        mode: Optional[ModelMode] = ModelMode.UPDATE_TASK_NN,
        **kwargs: Any,
    ) -> Dict:

        if mode == ModelMode.UPDATE_TASK_NN:
            results = self.forward_on_tasknn(x, labels, **kwargs)
        elif mode == ModelMode.UPDATE_SCORE_NN:
            results = self.forward_on_scorenn(x, labels, **kwargs)
        elif mode == ModelMode.COMPUTE_SCORE:
            score = self.compute_score(x, labels, **kwargs)
            results = {"score": score}
        elif mode is None:
            results = self.forward_on_tasknn(x, labels, **kwargs)
        else:
            raise ValueError

        return results

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        non_metrics: Dict[str, Union[float, int]] = self.get_all(
            reset=reset, type_=(LoggedScalarScalar,)
        )
        metrics = self.get_true_metrics(reset=reset)

        return {**metrics, **non_metrics}

    def forward_on_tasknn(  # type: ignore
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        meta: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        if meta is None:
            meta = {}
        results: Dict[str, Any] = {}
        # sampler needs one-hot labels of shape (batch, ...)

        if labels is not None and "labeled" in buffer.get('task'):
            labels = self.convert_to_one_hot(labels)
        elif "unlabeled" in buffer.get('task'):
            labels = labels.unsqueeze(-1)
            
        with self.inference_module.mode("inference"):
            y_pred, _, loss = self.inference_module(
                x, labels=labels, buffer=buffer
            )
        results["loss"] = loss
        results["y_pred"] = self.squeeze_y(y_pred)
        
        buffer["prob"] = y_pred.max(dim=-1)[0].mean(dim=-1)
        
        if labels is not None and "unlabeled" not in buffer.get('task'):
            self.calculate_metrics(
                x, labels, results["y_pred"], buffer, results
            )

        return results

    def compute_score(
        self,
        x: Any,
        y: torch.Tensor,  # (batch, num_samples or 1, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert self.score_nn is not None

        return self.score_nn(
            x, y, buffer, **kwargs
        )  # (batch, num_samples or 1)

    def forward_on_scorenn(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if labels is not None and "labeled" in buffer.get('task'):
            labels = self.convert_to_one_hot(labels)
        elif "unlabeled" in buffer.get('task'):
            labels = labels.unsqueeze(-1).expand(-1, -1, self.num_tags)

        # generate samples
        with torch.no_grad():
            with self.sampler.mode("sample"):
                y_hat, y_hat_extra, sampler_loss = self.sampler(
                    x, labels=labels, buffer=buffer
                )
        assert labels is not None
        
        results["score"] = buffer.get("score", float('-inf'))
        results["prob"] = buffer.get("prob", float('-inf'))
        
        if "unlabeled" in buffer.get('task') and self.use_pseudo_labeling:
            x_conf, labels_conf, y_hat_conf, buffer_conf = self.pseudo_labeling(x, labels, y_hat, buffer)
            # NCE ranking 
            loss = self.loss_fn(
                x_conf, self.unsqueeze_labels(labels_conf), y_hat_conf, y_hat_extra, buffer_conf
            )
        else:    
            # NCE ranking 
            loss = self.loss_fn(
                x, self.unsqueeze_labels(labels), y_hat, y_hat_extra, buffer
            )
        
        if self.use_pseudo_labeling:
            loss *= self.loss_fn.loss_weights.get(buffer.get('task')[0])
        
        results["y_hat"] = y_hat
        results["y_hat_extra"] = y_hat_extra
        results["loss"] = loss
    
        return results
    
    def pseudo_labeling(
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError
    
    def select_confident(
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError
    