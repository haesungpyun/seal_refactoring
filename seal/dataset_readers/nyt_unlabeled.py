from typing import (
    Dict,
    List,
    Any,
    Iterator,
)
import sys
if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload

import logging
import json
from transformers.models.bert.tokenization_bert import BertTokenizer

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    Field,
    MetadataField,
    MultiLabelField,
    SequenceLabelField
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from .srl_unlabeled import (
    SrlUnlabeledReader,
    _convert_tags_to_wordpiece_tags,
    _convert_verb_indices_to_wordpiece_indices
)
    
from wcmatch import glob

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: TextField  #:
    labels: MultiLabelField  #: types


@DatasetReader.register("nyt-unlabeled")
class NytUnlabeledReader(SrlUnlabeledReader):
    """
    SRL Reader for the New York times dataset
    """
    
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        domain_identifier: str = None,
        bert_model_name: str = None,
        max_length: int = None,
        **kwargs,
    ) -> None:
        super().__init__(
                manual_distributed_sharding=True,
                manual_multiprocess_sharding=True,
                **kwargs,
            )
        if token_indexers is not None:
            self._token_indexers = token_indexers
        elif bert_model_name is not None:
            from allennlp.data.token_indexers import PretrainedTransformerIndexer

            self._token_indexers = {"tokens": PretrainedTransformerIndexer(bert_model_name, max_length=max_length)}
        else:
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False
        
        self._max_length = max_length
        
    def _read(self, file_path: str) -> Iterator[Instance]:
        """Reads a datafile to produce instances

        Args:
            file_path: TODO

        Yields:
            data instances

        """
        
        for file_ in glob.glob(file_path, flags=glob.EXTGLOB):
            # logger.info(f"Reading {file_}")
            with open(file_) as f:
                for line in self.shard_iterable(f):
                    example = json.loads(line)          
                    instance = self.text_to_instance(**example)
                    if len(instance.fields['tokens'].tokens) > self._max_length:
                        continue  
                    yield instance
    
    
    def text_to_instance(  # type:ignore
        self,
        text: str,
        title: str,
        labels: List[str],
        general_descriptors: List[str],
        label_paths: List[List[str]],
        xml_path: str,
        taxonomy: List[str],
        **kwargs: Any
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            text: One line summary of article,
            title: Title of the article
            labels:list of labels,
            general_descriptors: Extra descriptors,
            label_path: List of taxonomies,
            xml_path: path to xml file,
            taxonomy: Taxonomy extracted form xml file
            **kwargs: Any

        Returns:
             :class:`Instance` of data

        """
    
        tokens = [Token(t) for t in text.split()]
        tags = ["O" for _ in tokens]
        verb_label = [0 for _ in tokens]
                        
        metadata_dict: Dict[str, Any] = {}
        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(
                [t.text for t in tokens]
            )
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            text_field = TextField(
                [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                token_indexers=self._token_indexers,
            )
            verb_indicator = SequenceLabelField(new_verbs, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {}
        fields["tokens"] = text_field
        fields["verb_indicator"] = verb_indicator

        if all(x == 0 for x in verb_label):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index
        metadata_dict['data_type'] = 'unlabeled'
        
        if tags:
            if self.bert_tokenizer is not None:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                unlabel_tags = [-100 for _ in range(len(new_tags))]
            else:
                unlabel_tags = [-100 for _ in range(len(tags))]    
            metadata_dict["gold_tags"] = tags
            fields["tags"] = SequenceLabelField(unlabel_tags, text_field)

        fields["metadata"] = MetadataField(metadata_dict)
        
        return Instance(fields)