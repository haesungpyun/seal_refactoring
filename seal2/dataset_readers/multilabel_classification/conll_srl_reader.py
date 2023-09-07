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
from wcmatch import glob

if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload

import logging
import json
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    Field,
    ArrayField,
    ListField,
    MetadataField,
    MultiLabelField,
    SequenceLabelField
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import Token


logger = logging.getLogger(__name__)


@DatasetReader.register("conll_srl")
class ConllSRLReader(DatasetReader):
    """
    Multi-label classification `dataset <https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html>`_.

    The output of this DatasetReader follows :class:`MultiInstanceEntityTyping`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        **kwargs: Any,
    ) -> None:
        """
        Arguments:
            tokenizer: The tokenizer to be used.
            token_indexers: The token_indexers to be used--one per embedder. Will usually be only one.
            **kwargs: Parent class args.
                `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py>`_

        """
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers


    def text_to_instance(  # type:ignore
        self,
        domain: str, 
        doc_id: str,
        words: List[Token],
        labels: List[str],
        verb_label:List[int],
        meta: Dict = None,
        **kwargs: Any,
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            title: Title of the book
            body: Blurb Text
            topics: Labels/Genres of the book
            idx: Identification number
            **kwargs: unused

        Returns:
             :class:`Instance` of data

        """
        meta_dict: Dict = {}
        
        wordpieces = []
        start_offsets = []
        end_offsets = []
        cumulative = 0

        new_verb_labels = []
        new_labels = []

        for idx, word in enumerate(words):
            token = word.text.lower()
            word_pieces = self._tokenizer.tokenize(token)
            
            start_offsets.append(cumulative+1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            
            _, label = labels[idx].split("-",1)
            verb = verb_label[idx]
            
            new_labels.insert(cumulative, labels[idx])
            new_verb_labels.insert(cumulative, verb_label[idx])
            
            for i in range(start_offsets[idx],end_offsets[idx]):
                new_labels.insert(i, "I-"+label)
                new_verb_labels.insert(i, verb)
            
            wordpieces.extend(word_pieces)
        
        wordpieces = ["[CLS]"] + wordpieces + ["[SEP]"]
        new_verb_labels = [0] + new_verb_labels + [0]
        new_labels = ["O" ] + new_labels + ["O"]
        
        text_field = TextField(
                [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                token_indexers=self._token_indexers,
            )
        verb_indicator = SequenceLabelField(new_verb_labels, text_field)

        fields: Dict[str, Field] = {}
        fields["tokens"] = text_field
        fields["verb_indicator"] = verb_indicator

        if all(x == 0 for x in verb_label):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = words[verb_index].text

        meta["words"] = [x.text for x in words]
        meta["verb"] = verb
        meta["verb_index"] = verb_index

        if labels is not None:
            fields["tags"] = SequenceLabelField(new_labels, text_field)
            meta["gold_tags"] = labels

        fields["metadata"] = MetadataField(meta)
        
        return Instance(fields)
    

    def _read(self, file_path: str) -> Iterator[Instance]:
        """Reads a datafile to produce instances

        Args:
            file_path: TODO

        Yields:
            data instances

        """
        import ast
        
        for file_ in glob.glob(file_path, flags=glob.EXTGLOB):
            # logger.info(f"Reading {file_}")
            with open(file_) as f:
                for line in self.shard_iterable(f):
                    example = json.loads(line)
                    example['words'] = ast.literal_eval(example['words'])
                    example['words'] = [Token(t) for t in example['words']]
                    example['labels'] = ast.literal_eval(example['labels'])
                    example['verbs'] = ast.literal_eval(example['verbs'])
                    instance = self.text_to_instance(**example)
                    yield instance
    

    def apply_token_indexers(self, instance: Instance) -> None:
        text_field = cast(TextField, instance.fields["x"])  # no runtime effect
        text_field._token_indexers = self._token_indexers
