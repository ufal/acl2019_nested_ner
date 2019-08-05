#!/bin/bash
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# This script evaluates the TensorFlow output, both during training and
# prediction phase, for nested corpora, using the evaluation script
# compare_nested_entities.py.

set -e

name="$1"
gold_dir="$2"

cat ${name}_system_predictions.conll | $(dirname $0)/conll2eval_nested.py > ${name}_system_entities.txt
$(dirname $0)/compare_nested_entities.py $(dirname $0)/${gold_dir}/${name}_gold_entities.txt ${name}_system_entities.txt > ${name}.eval
