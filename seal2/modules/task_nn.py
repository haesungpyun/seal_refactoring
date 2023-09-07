from typing import List, Tuple, Union, Dict, Any, Optional, Callable
from collections import defaultdict

from allennlp.nn.util import get_text_field_mask
from allennlp.data import Vocabulary, TextFieldTensors
import torch


class TaskNN(torch.nn.Module):
    """Base class for creating feature representation for any task.

    Inheriting classes should override the `foward` method.
    """

    pass


class CostAugmentedLayer(torch.nn.Module):
    def forward(self, inp: torch.Tensor, buffer: Dict) -> torch.Tensor:
        raise NotImplementedError

from allennlp.models.basic_classifier import BasicClassifier
class TextEncoder(torch.nn.Module):
    """Base class for creating feature representation for tasks with textual input.

    See `BasicClassifier <https://github.com/allenai/allennlp/blob/v2.5.0/allennlp/models/basic_classifier.py>`_ for reference.
    """

    default_implementation = "text-encoder"

    @staticmethod
    def generate_renamer(
        rename_map: Optional[Dict[str, Dict[str, str]]]
    ) -> Callable[
        [Dict[str, Dict[str, torch.Tensor]]],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        if rename_map is not None:
            assert rename_map is not None

            def renamer(
                field: Dict[str, Dict[str, torch.Tensor]]
            ) -> Dict[str, Dict[str, torch.Tensor]]:

                output: Dict[str, Dict[str, torch.Tensor]] = defaultdict(
                    lambda: {}
                )

                for indexer_name, rename_dict in rename_map.items():

                    for original, new in rename_dict.items():
                        output[indexer_name][new] = field[indexer_name][
                            original
                        ]

                return output

        else:

            def renamer(
                field: Dict[str, Dict[str, torch.Tensor]]
            ) -> Dict[str, Dict[str, torch.Tensor]]:

                return field

        return renamer

    def __init__(
        self,
        text_field_embedder: torch.nn.Module,
        seq2vec_encoder: torch.nn.Module,
        seq2seq_encoder: torch.nn.Module = None,
        feedforward: Optional[torch.nn.Module] = None,
        final_dropout: Optional[float] = None,
        text_field_embedder_rename_map: Optional[
            Dict[str, Dict[str, str]]
        ] = None,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.feedforward = feedforward
        self.text_field_embedder_rename_map = text_field_embedder_rename_map
        self.rename = TextEncoder.generate_renamer(
            self.text_field_embedder_rename_map
        )

        if final_dropout:
            self.final_dropout: Optional[torch.nn.Module] = torch.nn.Dropout(
                final_dropout
            )
        else:
            self.final_dropout = None

        self._output_dim = get_layer_output_dim(self.seq2vec_encoder, "out_features")
            
        if self.feedforward:
            self._output_dim = get_layer_output_dim(self.feedforward, "out_features")

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: TextFieldTensors) -> torch.Tensor:
        """
        Encodes the text input into a feature vector.
        """
        embedded_text = self.text_field_embedder(self.rename(x))
        mask = get_text_field_mask(x)

        if self.seq2seq_encoder:
            embedded_text = self.seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self.seq2vec_encoder(embedded_text, mask=mask)

        if self.final_dropout:
            embedded_text = self.final_dropout(embedded_text)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        return embedded_text

def get_layer_output_dim(
    layer: torch.nn.Module,
    feature_name: str = "out_features"
):
    if hasattr(layer, "get_output_dim"):
        return layer.get_output_dim()
    hidden_dims = list(map(lambda x: getattr(x, feature_name, None), list(layer.children())))
    for dim in hidden_dims[::-1]:
        if dim is not None:
            return dim            
