from typing import (
    Dict,
    List,
    Union,
    Any,
    Iterator,
    Tuple,
    cast,
    Optional,
    Iterable,
)
import sys
import itertools
if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload

import logging
import json
import dill
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    Field,
    ArrayField,
    ListField,
    MetadataField,
    MultiLabelField,
    SequenceLabelField,
    LabelField,
    
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import Token
from .wos_reader import WosReader
import glob

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: TextField  #:
    labels: LabelField  #: types


@DatasetReader.register("wos-keyword")
class WosKeywordReader(WosReader):
    """
    Reader for the Web of Science dataset

    """

    def example_to_fields(
        self,
        Abstract: str,
        target: List[str],
        keyword:List[str],
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        """Converts a dictionary containing an example datapoint to fields that can be used
        to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
        the meta dict.

        Args:
            Abstract,
            traget:list of labels,
            keyword:list of keyword,
            **kwargs: Any
        Returns:
            Dictionary of fields with the following entries:
                sentence: contains the body.
                mention: contains the title.
        """

        if meta is None:
            meta = {}

        meta["text"] = Abstract
        
        label = keyword[1] if (len(keyword) > 1) & (len(keyword[0]) == 1) else keyword[0]
        
        x = TextField(self._tokenizer.tokenize(Abstract))
        labels = LabelField(label)
        
        return {
            "x": x,
            "labels": labels,
        }