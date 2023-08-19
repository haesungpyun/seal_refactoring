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
from .blurb_genre_collection import BlurbGenreReader
import glob

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: TextField  #:
    labels: MultiLabelField  #: types


@DatasetReader.register("wos")
class WosReader(BlurbGenreReader):
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
        meta["labels"] = target
        meta["keyword"] = keyword
     
        x = TextField(self._tokenizer.tokenize(Abstract))
        labels = MultiLabelField(target)
        
        return {
            "x": x,
            "labels": labels,
        }

    def text_to_instance(  # type:ignore
        self,
        Abstract: str,
        target: List[str],
        keyword:List[str],
        **kwargs: Any
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            text: One line summary of article,
            title: Title of the article
            labels:list of labels,
            y: traget value,    
            yl1: y label 1,
            yl2: y label 2
            **kwargs: Any

        Returns:
             :class:`Instance` of data

        """
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(
            Abstract, target, keyword, meta=meta_dict
        )

        return Instance(
            {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
        )
  

@DatasetReader.register("wos-keyword-mlc")
class WosKeywordMLCReader(WosReader):
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
        meta["labels"] = keyword
     
        x = TextField(self._tokenizer.tokenize(Abstract))
        labels = MultiLabelField(keyword)
        
        return {
            "x": x,
            "labels": labels,
        }
      