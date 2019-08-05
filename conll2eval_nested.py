#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Creates an evaluation file with named entities.

Input: CoNLL file with linearized (encoded) nested named entity labels
delimited with |.

Output: One entity mention per line, two columns per line separated by table.
First column are entity mentino token ids separated by comma, second column is
a BIO or BILOU label.

The output can be then evaluated with compare_nested_entities.py.
""" 

import sys

COL_SEP = "\t"

def raw(label):
    return label[2:]

def flush(running_ids, running_forms, running_labels):
    for i in range(len(running_ids)):
        print(running_ids[i] + COL_SEP + running_labels[i] + COL_SEP + running_forms[i])
    return ([], [], [])

if __name__ == "__main__":

    i = 0
    running_ids = []
    running_forms = []
    running_labels = []
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        if not line: # flush entities
            (running_ids, running_forms, running_labels) = flush(running_ids, running_forms, running_labels)
        else:
            form , _, _, ne = line.split("\t")
            if ne == "O": # flush entities
                (running_ids, running_forms, running_labels) = flush(running_ids, running_forms, running_labels)
            else:
                labels = ne.split("|")
                for j in range(len(labels)): # for each label
                    label = labels[j]
                    if j < len(running_ids): # running entity
                        # previous running entity ends here, print and insert new entity instead
                        if label.startswith("B-") or label.startswith("U-") or running_labels[j] != raw(label):
                            print(running_ids[j] + COL_SEP + running_labels[j] + COL_SEP + running_forms[j])
                            running_ids[j] = str(i)
                            running_forms[j] = form
                        # entity continues, append ids and forms
                        else:
                            running_ids[j] += "," + str(i)
                            running_forms[j] += " " + form
                        running_labels[j] = raw(label)
                    else: # no running entities, new entity starts here, just append
                        running_ids.append(str(i))
                        running_forms.append(form)
                        running_labels.append(raw(label))
                        
        i += 1
