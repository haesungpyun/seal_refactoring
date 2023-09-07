# import torch, gc
# gc.collect()
# torch.cuda.empty_cache()
# from datetime import datetime
# import os
# import zipfile

# import os 

# import json
# import sys, os
# import pandas as pd
from datasets import load_dataset
import os
ds = load_dataset("jsonl", data_files='./data/bgc/train-fold-0.jsonl')


raise ValueError
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sent = "i am driving"
token = tokenizer(sent, return_tensor='pt')
# index = tokenizer(token)

raise ValueError

import os
i = 0
for ogs, atops in zip(os.walk('seal_og'), os.walk('seal')):
    if '.py' in str(atops[-1]) or '.py' in str(ogs[-1]):
        if '.pyc' in str(atops[-1]):
            continue
        for og_file, atops_file in zip(ogs[-1], atops[-1]):
            og_f = open(os.path.join(ogs[0], og_file), 'rb')
            a_f = open(os.path.join(atops[0], atops_file), 'rb')
            og_lines = og_f.readlines()
            at_lines = a_f.readlines()
            
            if str(og_lines)!=str(at_lines):
                print('\ndiff', os.path.join(ogs[0], og_file), os.path.join(atops[0], atops_file))

            for i, (o, a) in enumerate(zip(og_lines, at_lines)):
                if o != a:
                    # print('diff', os.path.join(ogs[0], og_file), i, os.path.join(atops[0], atops_file), i)
                    # print(o, a)
                    # print(i, end=' ')
                    ...

                


raise ValueError
inputio = sys.argv
print(inputio)
dir_path = inputio[1]
    
import os
from collections import Counter

# files = sorted(os.listdir(dir_path + '/scores'))
# cnt = Counter()
# for i, file in enumerate(files):
#     if not 'training' in file:
#         continue
#     f = open(dir_path + '/scores' + '/' + file, 'r')
#     cnt[len(f.readlines())] +=1
# print(dict(cnt))


dirs = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if 'srl_' in i]
error_dir = {}
for dir_path in dirs:
    dic = {}
    if not os.path.exists(os.path.join(dir_path, 'scores')):
        continue
    files = sorted(os.listdir(dir_path + '/scores'))
    for i, file in enumerate(files):
        if not 'training' in file:
            continue
        f = open(dir_path + '/scores' + '/' + file, 'r')
        dic[file] = len(f.readlines())
    cnt = Counter(dic.values())
    uncommons = cnt.most_common()[1:]
    print(dir_path, cnt)
    for value, _ in uncommons:
        error_dir[dir_path] = ([k for k, v in dic.items() if v == value], value, dict(cnt))
        # print(dir_path, [k for k, v in dic.items() if v == value], value)

from pathlib import Path
for key in error_dir:
    print(key)



# dirs = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if 'srl_' in i]
# error_dir = {}
# for dir_path in dirs:
#     dic = {}
#     if not os.path.exists(os.path.join(dir_path, 'scores')):
#         continue
#     files = sorted(os.listdir(dir_path + '/scores'))
#     for i, file in enumerate(files):
#         if not 'training' in file:
#             continue
#         f = open(dir_path + '/scores' + '/' + file, 'r')
#         dic[file] = len(f.readlines())
#     cnt = Counter(dic.values())
#     uncommons = cnt.most_common()[1:]
#     print(dir_path, cnt)
#     for value, _ in uncommons:
#         error_dir[dir_path] = ([k for k, v in dic.items() if v == value], value, dict(cnt))
#         # print(dir_path, [k for k, v in dic.items() if v == value], value)
# print(error_dir)


"""from allennlp_models.structured_prediction.dataset_readers.srl import SrlReader

from seal.dataset_readers.multilabel_classification.aapd_reader import AAPDReader
config = {
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "bert-base-uncased",
            "max_length": 512
            
        },
        "token_indexers": {
            "x": {
                "type": "pretrained_transformer",
                "model_name": "bert-base-uncased"
            }
        }        
    }
cofig = str(config)
from allennlp.common.params import Params
param = Params(config)
aapd_reader = AAPDReader.from_params(params=param)

tr_path = "/home/jylab_intern001/seal_ex/data/Castor-data-master/datasets/AAPD/data/train-fold-@(0|1|2|3|4|5|6|7|8|9).jsonl"
vl_path = "/home/jylab_intern001/seal_ex/data/Castor-data-master/datasets/AAPD/data/valid.jsonl"

tr_data = list(aapd_reader.read(tr_path))
vl_data = list(aapd_reader.read(vl_path))

from allennlp.data.vocabulary import Vocabulary

vocab = Vocabulary.from_instances(tr_data + vl_data)
print(vocab)

vocab.save_to_files("./vocabulary")


nyt_reader = NytUnlabeledReader(bert_model_name="bert-base-uncased",max_length=512)

print(sum(1 for _ in (nyt_reader.read(root))))

srl_reader = SrlReader()
root = "/home/jylab_intern001/seal_ex/data/conll-2012/v12/data/test/data/english/annotations"

for folder in os.listdir(root):
    print(folder)
    data_path = os.path.join(root, folder)
    data = list(srl_reader.read(data_path))
    print(len(data))

train_data_path = "/home/jylab_intern001/seal_ex/data/conll-2012/v12/data/train/data/english/annotations"
valid_data_path = "/home/jylab_intern001/seal_ex/data/conll-2012/v12/data/development/data/english/annotations"
validation_data_path = "/home/jylab_intern001/seal_ex/data/conll-2012/v12/data/development/data/english/annotations/",


tr_data = list(srl_reader.read(train_data_path))
vl_data = list(srl_reader.read(valid_data_path))
print(len(tr_data))
from allennlp.data.vocabulary import Vocabulary

vocab = Vocabulary.from_instances(tr_data + vl_data)
print(vocab)

vocab.save_to_files("./vocabulary")
"""