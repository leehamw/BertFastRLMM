import os
from pytorch_pretrained_bert import BertModel, BertConfig, BertTokenizer


# edit bert config
bert_cache_dir = './bert_cache'
tokenizer_cache_dir = './tokenizer_cache'
bert_type = './bert'


if not os.path.isdir(bert_cache_dir):
    os.mkdir(bert_cache_dir)
if not os.path.isdir(tokenizer_cache_dir):
    os.mkdir(tokenizer_cache_dir)


model = BertModel(bert_type, cache_dir=bert_cache_dir)
tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=tokenizer_cache_dir)
print('Cache files have been done...')