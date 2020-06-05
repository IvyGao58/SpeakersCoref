#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import util
import GCNModel as model
import ujson as json

if __name__ == "__main__":
    test_data = list()
    config = util.initialize_from_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print('Start to process data...')
    with open(config["test_path"], 'r') as f:
        for line in f:
            tmp_example = json.loads(line)
            test_data.append(tmp_example)

    print('finish processing data')

    mode = 'test'
    title_path = config["title_map_path"] if mode == 'test' else config["predict_title_map_path"]

    title_map = dict()
    with open(title_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            arr = line.split(': ')
            title_map[arr[0]] = arr[1].strip('\n')

    model = model.KnowledgePronounCorefModel(config)

    with tf.Session() as session:
        model.restore(session)
        model.evaluate(session, test_data, official_stdout=True, mode=mode, title_map=title_map)
        # model.evaluate(session)
        # model.evaluate_baseline_methods(session)

print('end')
