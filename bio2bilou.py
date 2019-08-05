#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Converts CoNLL file from BIO to BILOU encoding."""

import sys

def print_entity(lines):
    n = len(lines)
    if n > 0:
        if n == 1:
            lines[0][3] = lines[0][3].replace("I-","U-")
            lines[0][3] = lines[0][3].replace("B-","U-")
        else:
            lines[0][3] = lines[0][3].replace("I-", "B-")
            lines[n-1][3] = lines[n-1][3].replace("I-","L-")
        for i in range(n):
            print("\t".join(lines[i]))

if __name__ == "__main__":
    import argparse

    lines = []
    prev_label = "O"
    i = 0
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        i += 1
        if not line:
            print_entity(lines)
            lines = []
            prev_label = "O"
            print()
        else:
            if len(line.split("\t")) != 4:
                print("Incorrect line number " + str(i))
                sys.exit(1)
            form, lemma, tag, label = line.split("\t")
            # no entity, entity may have ended on previous lines
            if label == "O":
                print_entity(lines)
                lines = []
                print("\t".join([form, lemma, tag, label]))
            # new entity starts here, entity may have ended on previous lines
            elif label[-2:] != prev_label[-2:]:
                print_entity(lines)
                lines = []
                lines.append([form, lemma, tag, label])
            # other
            else:
                lines.append([form, lemma, tag, label])
            prev_label = label
