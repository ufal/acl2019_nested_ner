#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Converts CoNLL file from BILOU to BIO encoding."""

import sys


if __name__ == "__main__":
    import argparse

    lines = []
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        if not line:
            print()
        else:
            form, lemma, tag, label = line.split("\t")
            if label.startswith("U-"):
                label = label.replace("U-", "B-")
            if label.startswith("L-"):
                label = label.replace("L-", "I-")
            print("\t".join([form, lemma, tag, label]))
