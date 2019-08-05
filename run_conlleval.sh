#!/bin/bash
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# This script evaluates the TensorFlow output, both during training and
# prediction phase, for flat corpora (CoNLL-2003 and CoNLL-2002), using the
# official distributed evaluation script conlleval.

set -e

name="$1"
gold="$2"
system="$3"

if [ $name == "dev" ]; then
  $(dirname $0)/bilou2bio.py < ${system} > ${name}_system_bio.conll
  $(dirname $0)/bilou2bio.py < $(dirname $0)/${gold} > ${name}_gold_bio.conll
  paste ${name}_gold_bio.conll ${name}_system_bio.conll | cut -f1,2,3,4,8 > ${name}_conlleval_input.conll
elif [ $name == "test" ]; then
  $(dirname $0)/bilou2bio.py < ${system} > ${name}_system_bio.conll
  paste $(dirname $0)/${gold} ${name}_system_bio.conll | cut -f1,2,3,4,8 > ${name}_conlleval_input.conll
else
  echo "./run_conlleval.sh: Unknown file name \"$name\"."
  exit 1
fi

$(dirname $0)/conlleval -d "\t" < ${name}_conlleval_input.conll > $name.eval
