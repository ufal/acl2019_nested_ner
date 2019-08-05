#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Evaluates nested entity predictions.

The predictions are supposed to be in the following format:

One entity mention per line, two columns per line separated by table. First
column are entity mention token ids separated by comma, second column is
a BIO or BILOU label. Only classes are compared, the B-, I-, L- and U-
prefixes are stripped.
"""


import sys


if __name__ == "__main__":

    with open(sys.argv[1], "r", encoding="utf-8") as fr:
        gold_entities = fr.readlines()
        for i in range(len(gold_entities)):
            gold_entities[i] = gold_entities[i].split("\t")[:2]

    with open(sys.argv[2], "r", encoding="utf-8") as fr:
        system_entities = fr.readlines()
        for i in range(len(system_entities)):
            system_entities[i] = system_entities[i].split("\t")[:2]

    correct_retrieved = 0
    for entity in system_entities:
        if entity in gold_entities:
            correct_retrieved += 1

    recall = correct_retrieved / len(gold_entities) if gold_entities else 0
    precision = correct_retrieved / len(system_entities) if system_entities else 0
    f1 = (2 * recall * precision) / (recall + precision) if recall+precision else 0

    print("Correct retrieved: {}".format(correct_retrieved))
    print("Retrieved: {}".format(len(system_entities)))
    print("Gold: {}".format(len(gold_entities)))
    print("Recall: {:.2f}".format(recall*100))
    print("Precision: {:.2f}".format(precision*100))
    print("F1: {:.2f}".format(f1*100))
