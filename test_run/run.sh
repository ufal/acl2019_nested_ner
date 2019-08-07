#!/bin/bash
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Tagger test run with minimal parameters.

set -e

cat test.conll | ../conll2eval_nested.py > test_gold_entities.txt

# Seq2seq
(cd ../ && ./tagger.py --corpus=ACE2004 --train_data=test_run/train.conll --test_data=test_run/test.conll --decoding=seq2seq --epochs=50:1e-3,8:1e-4 --name=test_run)

# LSTM-CRF
(cd ../ && ./tagger.py --corpus=ACE2004 --train_data=test_run/train.conll --test_data=test_run/test.conll --decoding=CRF --epochs=50:1e-3,8:1e-4 --name=test_run)
