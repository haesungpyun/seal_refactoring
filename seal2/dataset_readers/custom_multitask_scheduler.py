from collections import defaultdict
from typing import Any, Dict, Iterable, Union, List, Mapping

import more_itertools

from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance
from allennlp.data.data_loaders.multitask_scheduler import (
    MultiTaskScheduler,
    HomogeneousRoundRobinScheduler,
    _chunked_iterator
)

from itertools import cycle

@MultiTaskScheduler.register("custom_roundrobin")
class CustomRoundRobinScheduler(HomogeneousRoundRobinScheduler):
    def __init__(self, trunc_task: str = 'labeled', **kwargs: Any):
        super().__init__(**kwargs)
        self._trunc_task = trunc_task
    
    def batch_instances(
        self, epoch_instances: Dict[str, Iterable[Instance]]
    ) -> Iterable[List[Instance]]:
        chunked_iterators: Dict[str, Iterable[Instance]]= {
            dataset: _chunked_iterator(iterator, self.batch_size[dataset], self.drop_last)
            for dataset, iterator in epoch_instances.items()
        }
        return custom_roundrobin(chunked_iterators, self._trunc_task)

    def count_batches(self, dataset_counts: Dict[str, int]) -> int:
        result = 0
        for dataset, count in dataset_counts.items():
            batch_size = self.batch_size[dataset]
            result += count // batch_size
            if not self.drop_last and count % batch_size != 0:
                result += 1
        return result

def custom_roundrobin(iter_dict, trunk_task='labeled'):
    """Yields an item from each iterable, alternating between them.
        >>> list(custom_roundrobin({'alphabets':'ABCDE', 'ints':'12'}))
        ['A', '1', 'B', '2', 'C', '1', 'D', '2', 'E']
    """
    if len(iter_dict) == 1:
        trunk_task = list(iter_dict.keys()).pop()
    
    nexts = {
        key: iter(it).__next__
        if key == trunk_task else cycle(it).__next__ 
        for key, it in iter_dict.items()
    }
    next_keys = cycle(nexts.keys()).__next__
    while True:
        key = next_keys()
        try:
            next = nexts.get(key)
            yield next()
        except StopIteration:
            break
