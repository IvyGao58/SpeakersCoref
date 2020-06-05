#!/usr/bin/env python
# encoding: utf-8
'''
time: 2020/4/29 10:41
@desc: configuration for data processing
'''

official_status = '../data/raw/official_status.txt'
annotate_data_path = '../data/raw/annotate.xls'
stop_words_file = '../data/raw/stopwords.txt'
context_json_path = '../data/raw/context.json'
downer_data_path = '../data/raw/downer_context.json'

# file for P-S mappings
train_p2sSort_jsonlines = '../data/law/p2s/train.p2sSort.jsonlines'
dev_p2sSort_jsonlines = '../data/law/p2s/dev.p2sSort.jsonlines'
test_p2sSort_jsonlines = '../data/law/p2s/test.p2sSort.jsonlines'

p2s_ns_mapping = '../data/predict/p2s/p2s_ns_mapping.jsonlines'
p2s_predict = '../data/predict/p2s/eval_output.log'
p2s_title_map_path = '../data/law/p2s/p2s_title_map.txt'

# file for P-N mappings
train_p2n_jsonlines = '../data/law/p2n/train.jsonlines'
dev_p2n_jsonlines = '../data/law/p2n/dev.jsonlines'
test_p2n_jsonlines = '../data/law/p2n/test.jsonlines'

p2n_ns_mapping = '../data/predict/p2n/p2n_ns_mapping.jsonlines'
p2n_predict = '../data/predict/p2n/eval_output.log'
p2n_title_map_path = '../data/law/p2n/p2n_title_map.txt'


