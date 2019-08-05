#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Nested NER training and evaluation in TensorFlow."""

import json
import os
import sys

import fasttext
import numpy as np
import tensorflow as tf
import word2vec

import morpho_dataset


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_forms, num_form_chars, num_lemmas, num_lemma_chars, num_pos,
                  pretrained_form_we_dim, pretrained_lemma_we_dim, pretrained_fasttext_dim,
                  num_tags, tag_bos, tag_eow, pretrained_bert_dim, pretrained_flair_dim, pretrained_elmo_dim,
                  predict_only):
        with self.session.graph.as_default():

            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.form_ids = tf.placeholder(tf.int32, [None, None], name="form_ids")
            self.lemma_ids = tf.placeholder(tf.int32, [None, None], name="lemma_ids")
            self.pos_ids = tf.placeholder(tf.int32, [None, None], name="pos_ids")
            self.pretrained_form_wes = tf.placeholder(tf.float32, [None, None, pretrained_form_we_dim], name="pretrained_form_wes")
            self.pretrained_lemma_wes = tf.placeholder(tf.float32, [None, None, pretrained_lemma_we_dim], name="pretrained_lemma_wes")
            self.pretrained_fasttext_wes = tf.placeholder(tf.float32, [None, None, pretrained_fasttext_dim], name="fasttext_wes")
            self.pretrained_bert_wes = tf.placeholder(tf.float32, [None, None, pretrained_bert_dim], name="bert_wes")
            self.pretrained_flair_wes = tf.placeholder(tf.float32, [None, None, pretrained_flair_dim], name="flair_wes")
            self.pretrained_elmo_wes = tf.placeholder(tf.float32, [None, None, pretrained_elmo_dim], name="elmo_wes")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            self.is_training = tf.placeholder(tf.bool, [])
            self.learning_rate = tf.placeholder(tf.float32, [])

            if args.including_charseqs:
                self.form_charseqs = tf.placeholder(tf.int32, [None, None], name="form_charseqs")
                self.form_charseq_lens = tf.placeholder(tf.int32, [None], name="form_charseq_lens")
                self.form_charseq_ids = tf.placeholder(tf.int32, [None,None], name="form_charseq_ids")
                
                self.lemma_charseqs = tf.placeholder(tf.int32, [None, None], name="lemma_charseqs")
                self.lemma_charseq_lens = tf.placeholder(tf.int32, [None], name="lemma_charseq_lens")
                self.lemma_charseq_ids = tf.placeholder(tf.int32, [None,None], name="lemma_charseq_ids")
                
            # RNN Cell
            if args.rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
            elif args.rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

            inputs = []

            # Trainable embeddings for forms
            form_embeddings = tf.get_variable("form_embeddings", shape=[num_forms, args.we_dim], dtype=tf.float32)
            inputs.append(tf.nn.embedding_lookup(form_embeddings, self.form_ids))
            
            # Trainable embeddings for lemmas
            lemma_embeddings = tf.get_variable("lemma_embeddings", shape=[num_lemmas, args.we_dim], dtype=tf.float32)
            inputs.append(tf.nn.embedding_lookup(lemma_embeddings, self.lemma_ids))
            
            # POS encoded as one-hot vectors
            inputs.append(tf.one_hot(self.pos_ids, num_pos))
            
            # Pretrained embeddings for forms
            if args.form_wes_model:
                inputs.append(self.pretrained_form_wes)

            # Pretrained embeddings for lemmas
            if args.lemma_wes_model:
                inputs.append(self.pretrained_lemma_wes)
            
            # Fasttext form embeddings
            if args.fasttext_model:
                inputs.append(self.pretrained_fasttext_wes)

            # BERT form embeddings
            if pretrained_bert_dim:
                inputs.append(self.pretrained_bert_wes)

            # Flair form embeddings
            if pretrained_flair_dim:
                inputs.append(self.pretrained_flair_wes)
                
            # ELMo form embeddings
            if pretrained_elmo_dim:
                inputs.append(self.pretrained_elmo_wes)

            # Character-level form embeddings
            if args.including_charseqs:

                # Generate character embeddings for num_form_chars of dimensionality args.cle_dim.
                character_embeddings = tf.get_variable("form_character_embeddings",
                                                        shape=[num_form_chars, args.cle_dim],
                                                        dtype=tf.float32)
                
                # Embed self.form_charseqs (list of unique form in the batch) using the character embeddings.
                characters_embedded = tf.nn.embedding_lookup(character_embeddings, self.form_charseqs)
                
                # Use tf.nn.bidirectional.rnn to process embedded self.form_charseqs
                # using a GRU cell of dimensionality args.cle_dim.
                _, (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(
                        tf.nn.rnn_cell.GRUCell(args.cle_dim), tf.nn.rnn_cell.GRUCell(args.cle_dim),
                        characters_embedded, sequence_length=self.form_charseq_lens, dtype=tf.float32, scope="form_cle")
                
                # Sum the resulting fwd and bwd state to generate character-level form embedding (CLE)
                # of unique forms in the batch.
                cle = tf.concat([state_fwd, state_bwd], axis=1)
                
                # Generate CLEs of all form in the batch by indexing the just computed embeddings
                # by self.form_charseq_ids (using tf.nn.embedding_lookup).
                cle_embedded = tf.nn.embedding_lookup(cle, self.form_charseq_ids)
                
                # Concatenate the form embeddings (computed above in inputs) and the CLE (in this order).
                inputs.append(cle_embedded)

            # Character-level lemma embeddings
            if args.including_charseqs:

                character_embeddings = tf.get_variable("lemma_character_embeddings",
                                                        shape=[num_lemma_chars, args.cle_dim],
                                                        dtype=tf.float32)
                characters_embedded = tf.nn.embedding_lookup(character_embeddings, self.lemma_charseqs)
                _, (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(
                        tf.nn.rnn_cell.GRUCell(args.cle_dim), tf.nn.rnn_cell.GRUCell(args.cle_dim),
                        characters_embedded, sequence_length=self.lemma_charseq_lens, dtype=tf.float32, scope="lemma_cle")
                cle = tf.concat([state_fwd, state_bwd], axis=1)
                cle_embedded = tf.nn.embedding_lookup(cle, self.lemma_charseq_ids)
                inputs.append(cle_embedded)

            # Concatenate inputs
            inputs = tf.concat(inputs, axis=2)
            
            # Dropout
            inputs_dropout = tf.layers.dropout(inputs, rate=args.dropout, training=self.is_training)
            
            # Computation
            hidden_layer_dropout = inputs_dropout # first layer is input
            for i in range(args.rnn_layers):
                (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
                    hidden_layer_dropout, sequence_length=self.sentence_lens, dtype=tf.float32,
                    scope="RNN-{}".format(i))
                hidden_layer = tf.concat([hidden_layer_fwd, hidden_layer_bwd], axis=2)
                if i == 0: hidden_layer_dropout = 0
                hidden_layer_dropout += tf.layers.dropout(hidden_layer, rate=args.dropout, training=self.is_training)

            # Decoders
            if args.decoding == "CRF": # conditional random fields
                output_layer = tf.layers.dense(hidden_layer_dropout, num_tags)
                weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                    output_layer, self.tags, self.sentence_lens)
                loss = tf.reduce_mean(-log_likelihood)
                self.predictions, viterbi_score = tf.contrib.crf.crf_decode(
                    output_layer, transition_params, self.sentence_lens)
                self.predictions_training = self.predictions
            elif args.decoding == "ME": # vanilla maximum entropy
                output_layer = tf.layers.dense(hidden_layer_dropout, num_tags)
                weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
                if args.label_smoothing:
                    gold_labels = tf.one_hot(self.tags, num_tags) * (1 - args.label_smoothing) + args.label_smoothing / num_tags
                    loss = tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights)
                else:
                    loss = tf.losses.sparse_softmax_cross_entropy(self.tags, output_layer, weights=weights)
                self.predictions = tf.argmax(output_layer, axis=2)
                elf.predictions_training = self.predictions
            elif args.decoding in ["LSTM", "seq2seq"]: # Decoder
                # Generate target embeddings for target chars, of shape [target_chars, args.char_dim].
                tag_embeddings = tf.get_variable("tag_embeddings", shape=[num_tags, args.we_dim], dtype=tf.float32)

                # Embed the target_seqs using the target embeddings. 
                tags_embedded = tf.nn.embedding_lookup(tag_embeddings, self.tags)

                decoder_rnn_cell = rnn_cell(args.rnn_cell_dim)

                # Create a `decoder_layer` -- a fully connected layer with
                # target_chars neurons used in the decoder to classify into target characters.
                decoder_layer = tf.layers.Dense(num_tags)
                
                sentence_lens = self.sentence_lens
                max_sentence_len = tf.reduce_max(sentence_lens)
                tags = self.tags
                # The DecoderTraining will be used during training. It will output logits for each
                # target character.
                class DecoderTraining(tf.contrib.seq2seq.Decoder):
                    @property
                    def batch_size(self): return tf.shape(hidden_layer_dropout)[0]
                    @property
                    def output_dtype(self): return tf.float32 # Type for logits of target characters
                    @property
                    def output_size(self): return num_tags # Length of logits for every output
                    @property
                    def tag_eow(self): return tag_eow

                    def initialize(self, name=None):
                        states = decoder_rnn_cell.zero_state(self.batch_size, tf.float32)
                        inputs = [tf.nn.embedding_lookup(tag_embeddings, tf.fill([self.batch_size], tag_bos)), hidden_layer_dropout[:,0]]
                        inputs = tf.concat(inputs, axis=1)
                        if args.decoding == "seq2seq":
                            predicted_eows = tf.zeros([self.batch_size], dtype=tf.int32)
                            inputs = (inputs, predicted_eows)
                        finished = sentence_lens <= 0
                        return finished, inputs, states

                    def step(self, time, inputs, states, name=None):
                        if args.decoding == "seq2seq":
                            inputs, predicted_eows = inputs
                        outputs, states = decoder_rnn_cell(inputs, states)
                        outputs = decoder_layer(outputs)
                        next_input = [tf.nn.embedding_lookup(tag_embeddings, tags[:,time])]
                        if args.decoding == "seq2seq":
                            predicted_eows += tf.to_int32(tf.equal(tags[:, time], self.tag_eow))
                            indices = tf.where(tf.one_hot(tf.minimum(predicted_eows, max_sentence_len - 1), tf.reduce_max(predicted_eows) + 1))
                            next_input.append(tf.gather_nd(hidden_layer_dropout, indices))
                        else:
                            next_input.append(hidden_layer_dropout[:,tf.minimum(time + 1, max_sentence_len - 1)])
                        next_input = tf.concat(next_input, axis=1)
                        if args.decoding == "seq2seq":
                            next_input = (next_input, predicted_eows)
                            finished = sentence_lens <= predicted_eows
                        else:
                            finished = sentence_lens <= time + 1
                        return outputs, states, next_input, finished
                output_layer, _, prediction_training_lens = tf.contrib.seq2seq.dynamic_decode(DecoderTraining())
                self.predictions_training = tf.argmax(output_layer, axis=2, output_type=tf.int32)
                weights = tf.sequence_mask(prediction_training_lens, dtype=tf.float32)
                if args.label_smoothing:
                    gold_labels = tf.one_hot(self.tags, num_tags) * (1 - args.label_smoothing) + args.label_smoothing / num_tags
                    loss = tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights)
                else:
                    loss = tf.losses.sparse_softmax_cross_entropy(self.tags, output_layer, weights=weights)

                # The DecoderPrediction will be used during prediction. It will
                # directly output the predicted target characters.
                class DecoderPrediction(tf.contrib.seq2seq.Decoder):
                    @property
                    def batch_size(self): return tf.shape(hidden_layer_dropout)[0]
                    @property
                    def output_dtype(self): return tf.int32 # Type for predicted target characters
                    @property
                    def output_size(self): return 1 # Will return just one output
                    @property
                    def tag_eow(self): return tag_eow

                    def initialize(self, name=None):
                        states = decoder_rnn_cell.zero_state(self.batch_size, tf.float32)
                        inputs = [tf.nn.embedding_lookup(tag_embeddings, tf.fill([self.batch_size], tag_bos)), hidden_layer_dropout[:,0]]
                        inputs = tf.concat(inputs, axis=1)
                        if args.decoding == "seq2seq":
                            predicted_eows = tf.zeros([self.batch_size], dtype=tf.int32)
                            inputs = (inputs, predicted_eows) 
                        finished = sentence_lens <= 0
                        return finished, inputs, states
                    
                    def step(self, time, inputs, states, name=None):
                        if args.decoding == "seq2seq":
                            inputs, predicted_eows = inputs
                        outputs, states = decoder_rnn_cell(inputs, states)
                        outputs = decoder_layer(outputs)
                        outputs = tf.argmax(outputs, axis=1, output_type=self.output_dtype)
                        next_input = [tf.nn.embedding_lookup(tag_embeddings, outputs)]
                        if args.decoding == "seq2seq":
                            predicted_eows += tf.to_int32(tf.equal(outputs, self.tag_eow))
                            indices = tf.where(tf.one_hot(tf.minimum(predicted_eows, max_sentence_len - 1), tf.reduce_max(predicted_eows) + 1))
                            next_input.append(tf.gather_nd(hidden_layer_dropout, indices))
                        else:
                            next_input.append(hidden_layer_dropout[:,tf.minimum(time + 1, max_sentence_len - 1)])
                        next_input = tf.concat(next_input, axis=1)
                        if args.decoding == "seq2seq":
                            next_input = (next_input, predicted_eows)
                            finished = sentence_lens <= predicted_eows
                        else:
                            finished = sentence_lens <= time + 1
                        return outputs, states, next_input, finished
                self.predictions, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        DecoderPrediction(), maximum_iterations=3*tf.reduce_max(self.sentence_lens) + 10)
                
            # Saver
            self.saver = tf.train.Saver(max_to_keep=1)
            if predict_only: return

            # Training
            global_step = tf.train.create_global_step()
            self.training = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(loss, global_step=global_step)

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions_training, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            self.metrics = {}
            self.metrics_summarize = {}
            for metric in ["precision", "recall", "F1"]:
                self.metrics[metric] = tf.placeholder(tf.float32, [], name=metric)
                self.metrics_summarize[metric] = {}
                with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    for dataset in ["dev", "test"]:
                        self.metrics_summarize[metric][dataset] = tf.contrib.summary.scalar(dataset + "/" + metric,
                                                                                            self.metrics[metric])

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)


    def train_epoch(self, train, learning_rate, args):
        while not train.epoch_finished():
            seq2seq = args.decoding == "seq2seq"
            batch_dict = train.next_batch(args.batch_size, args.form_wes_model, args.lemma_wes_model, args.fasttext_model, including_charseqs=args.including_charseqs, seq2seq=seq2seq)
            if args.word_dropout:
                mask = np.random.binomial(n=1, p=args.word_dropout, size=batch_dict["word_ids"][train.FORMS].shape)
                batch_dict["word_ids"][train.FORMS] = (1 - mask) * batch_dict["word_ids"][train.FORMS] + mask * train.factors[train.FORMS].words_map["<unk>"]
                
                mask = np.random.binomial(n=1, p=args.word_dropout, size=batch_dict["word_ids"][train.LEMMAS].shape)
                batch_dict["word_ids"][train.LEMMAS] = (1 - mask) * batch_dict["word_ids"][train.LEMMAS] + mask * train.factors[train.LEMMAS].words_map["<unk>"]

            self.session.run(self.reset_metrics)
            feeds = {self.sentence_lens: batch_dict["sentence_lens"],
                     self.form_ids: batch_dict["word_ids"][train.FORMS],
                     self.lemma_ids: batch_dict["word_ids"][train.LEMMAS],
                     self.pos_ids: batch_dict["word_ids"][train.POS],
                     self.tags: batch_dict["word_ids"][train.TAGS],
                     self.is_training: True,
                     self.learning_rate: learning_rate}
            if args.form_wes_model: # pretrained form embeddings
                feeds[self.pretrained_form_wes] = batch_dict["batch_form_pretrained_wes"]
            if args.lemma_wes_model: # pretrained lemma embeddings
                feeds[self.pretrained_lemma_wes] = batch_dict["batch_lemma_pretrained_wes"]
            if args.fasttext_model: # fasttext form embeddings
                feeds[self.pretrained_fasttext_wes] = batch_dict["batch_form_fasttext_wes"]
            if args.bert_embeddings_train: # BERT embeddings
                feeds[self.pretrained_bert_wes] = batch_dict["batch_bert_wes"]
            if args.flair_train: # flair embeddings
                feeds[self.pretrained_flair_wes] = batch_dict["batch_flair_wes"]
            if args.elmo_train: # elmo embeddings
                feeds[self.pretrained_elmo_wes] = batch_dict["batch_elmo_wes"]

            if args.including_charseqs: # character-level embeddings
                feeds[self.form_charseqs] = batch_dict["batch_charseqs"][train.FORMS]
                feeds[self.form_charseq_lens] = batch_dict["batch_charseq_lens"][train.FORMS]
                feeds[self.form_charseq_ids] = batch_dict["batch_charseq_ids"][train.FORMS]
                
                feeds[self.lemma_charseqs] = batch_dict["batch_charseqs"][train.LEMMAS]
                feeds[self.lemma_charseq_lens] = batch_dict["batch_charseq_lens"][train.LEMMAS]
                feeds[self.lemma_charseq_ids] = batch_dict["batch_charseq_ids"][train.LEMMAS]

            self.session.run([self.training, self.summaries["train"]], feeds)


    def evaluate(self, dataset_name, dataset, args):
        with open("{}/{}_system_predictions.conll".format(args.logdir, dataset_name), "w", encoding="utf-8") as prediction_file:
            self.predict(dataset_name, dataset, args, prediction_file, evaluating=True)

        f1 = 0.0
        if args.corpus in ["CoNLL_en", "CoNLL_de", "CoNLL_nl", "CoNLL_es"]:
            os.system("cd {} && ../../run_conlleval.sh {} {} {}_system_predictions.conll".format(args.logdir, dataset_name, args.__dict__[dataset_name + "_data"], dataset_name))

            with open("{}/{}.eval".format(args.logdir,dataset_name), "r", encoding="utf-8") as result_file:
                for line in result_file:
                    line = line.strip("\n")
                    if line.startswith("accuracy:"):
                        f1 = float(line.split()[-1])
                        self.session.run(self.metrics_summarize["F1"][dataset_name], {self.metrics["F1"]: f1})

            return f1
       elif args.corpus in [ "ACE2004", "ACE2005", "GENIA" ]: # nested named entities evaluation
            os.system("cd {} && ../../run_eval_nested.sh {} {}".format(args.logdir, dataset_name, os.path.dirname(args.__dict__[dataset_name + "_data"])))

            with open("{}/{}.eval".format(args.logdir,dataset_name), "r", encoding="utf-8") as result_file:
                for line in result_file:
                    line = line.strip("\n")
                    if line.startswith("Recall:"):
                        recall = float(line.split(" ")[1])
                    if line.startswith("Precision:"):
                        precision = float(line.split(" ")[1])
                    if line.startswith("F1:"):
                        f1 = float(line.split(" ")[1])
                        for metric, value in [["precision", precision], ["recall", recall], ["F1", f1]]:
                            self.session.run(self.metrics_summarize[metric][dataset_name], {self.metrics[metric]: value})
            return f1
        else:
            raise ValueError("Unknown corpus {}".format(args.corpus))


    def predict(self, dataset_name, dataset, args, prediction_file, evaluating=False):
        if evaluating:
            self.session.run(self.reset_metrics)
        tags = []
        while not dataset.epoch_finished():
            seq2seq = args.decoding == "seq2seq"
            batch_dict = dataset.next_batch(args.batch_size, args.form_wes_model, args.lemma_wes_model, args.fasttext_model, args.including_charseqs, seq2seq=seq2seq)
            targets = [self.predictions]
            feeds = {self.sentence_lens: batch_dict["sentence_lens"],
                    self.form_ids: batch_dict["word_ids"][dataset.FORMS],
                    self.lemma_ids: batch_dict["word_ids"][train.LEMMAS],
                    self.pos_ids: batch_dict["word_ids"][train.POS],
                    self.is_training: False}
            if evaluating:
                targets.extend([self.update_accuracy, self.update_loss])
                feeds[self.tags] = batch_dict["word_ids"][dataset.TAGS]
            if args.form_wes_model: # pretrained form embeddings
                feeds[self.pretrained_form_wes] = batch_dict["batch_form_pretrained_wes"]
            if args.lemma_wes_model: # pretrained lemma embeddings
                feeds[self.pretrained_lemma_wes] = batch_dict["batch_lemma_pretrained_wes"]
            if args.fasttext_model: # fasttext form embeddings
                feeds[self.pretrained_fasttext_wes] = batch_dict["batch_form_fasttext_wes"]
            if args.bert_embeddings_dev or args.bert_embeddings_test: # BERT embeddings
                feeds[self.pretrained_bert_wes] = batch_dict["batch_bert_wes"]
            if args.flair_dev or args.flair_test: # flair embeddings
                feeds[self.pretrained_flair_wes] = batch_dict["batch_flair_wes"]
            if args.elmo_dev or args.elmo_test: # elmo embeddings
                feeds[self.pretrained_elmo_wes] = batch_dict["batch_elmo_wes"]

            if args.including_charseqs: # character-level embeddings
                feeds[self.form_charseqs] = batch_dict["batch_charseqs"][dataset.FORMS]
                feeds[self.form_charseq_lens] = batch_dict["batch_charseq_lens"][dataset.FORMS]
                feeds[self.form_charseq_ids] = batch_dict["batch_charseq_ids"][dataset.FORMS]
                
                feeds[self.lemma_charseqs] = batch_dict["batch_charseqs"][dataset.LEMMAS]
                feeds[self.lemma_charseq_lens] = batch_dict["batch_charseq_lens"][dataset.LEMMAS]
                feeds[self.lemma_charseq_ids] = batch_dict["batch_charseq_ids"][dataset.LEMMAS]

            tags.extend(self.session.run(targets, feeds)[0])

        if evaluating:
            self.session.run([self.current_accuracy, self.summaries[dataset_name]])
     
        forms = dataset.factors[dataset.FORMS].strings
        for s in range(len(forms)):
            j = 0
            for i in range(len(forms[s])):
                if args.decoding == "seq2seq": # collect all tags until <eow>
                    labels = []
                    while j < len(tags[s]) and dataset.factors[dataset.TAGS].words[tags[s][j]] != "<eow>":
                        labels.append(dataset.factors[dataset.TAGS].words[tags[s][j]])
                        j += 1
                    j += 1 # skip the "<eow>"
                    print("{}\t_\t_\t{}".format(forms[s][i], "|".join(labels)), file=prediction_file)
                else:
                    print("{}\t_\t_\t{}".format(forms[s][i], dataset.factors[dataset.TAGS].words[tags[s][i]]), file=prediction_file)
            print("", file=prediction_file)
        

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--bert_embeddings_dev", default=None, type=str, help="Pretrained BERT embeddings for dev data.")
    parser.add_argument("--bert_embeddings_test", default=None, type=str, help="Pretrained BERT embeddings for test data.")
    parser.add_argument("--bert_embeddings_train", default=None, type=str, help="Pretrained BERT embeddings for train data.")
    parser.add_argument("--beta_2", default=0.98, type=float, help="Beta 2.")
    parser.add_argument("--corpus", default="CoNLL_en", type=str, help="CoNLL_en|CoNLL_de|CoNLL_nl|CoNLL_es|ACE2004|ACE2005|GENIA.")
    parser.add_argument("--cle_dim", default=128, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--decoding", default="CRF", type=str, help="Decoding: [CRF|ME|LSTM|seq2seq].")
    parser.add_argument("--dev_data", default=None, type=str, help="Dev data.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
    parser.add_argument("--elmo_dev", default=None, type=str, help="ELMo dev embeddings.")
    parser.add_argument("--elmo_test", default=None, type=str, help="ELMo test embeddings.")
    parser.add_argument("--elmo_train", default=None, type=str, help="ELMo train embeddings.")
    parser.add_argument("--epochs", default="10:1e-3", type=str, help="Epochs and learning rates.")
    parser.add_argument("--fasttext_model", default=None, type=str, help="Fasttext subwords.")
    parser.add_argument("--flair_dev", default=None, type=str, help="Flair dev embeddings.")
    parser.add_argument("--flair_test", default=None, type=str, help="Flair test embeddings.")
    parser.add_argument("--flair_train", default=None, type=str, help="Flair train embeddings.")
    parser.add_argument("--form_wes_model", default=None, type=str, help="Pretrained form WEs.")
    parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
    parser.add_argument("--lemma_wes_model", default=None, type=str, help="Pretrained lemma WEs.")
    parser.add_argument("--max_sentences", default=None, type=int, help="Number of training sentences (for debugging).")
    parser.add_argument("--name", default=None, type=str, help="Experiment name.")
    parser.add_argument("--predict", default=None, type=str, help="Predict using the passed model.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=256, type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--test_data", default=None, type=str, help="Test data.")
    parser.add_argument("--train_data", default=None, type=str, help="Training data.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=256, type=int, help="Word embedding dimension.")
    parser.add_argument("--word_dropout", default=0.2, type=float, help="Word dropout.")
    args = parser.parse_args()

    if args.predict:
        # Load saved options from the model
        with open("{}/options.json".format(args.predict), mode="r") as options_file:
            args = argparse.Namespace(**json.load(options_file))
        parser.parse_args(namespace=args)
    else:
        # Create logdir name
        logargs = dict(vars(args).items())
        logargs["form_wes_model"] = 1 if args.form_wes_model else 0
        logargs["lemma_wes_model"] = 1 if args.lemma_wes_model else 0
        del logargs["bert_embeddings_dev"]
        del logargs["bert_embeddings_test"]
        del logargs["bert_embeddings_train"]
        del logargs["beta_2"]
        del logargs["cle_dim"]
        del logargs["dev_data"]
        del logargs["dropout"]
        del logargs["elmo_dev"]
        del logargs["elmo_test"]
        del logargs["elmo_train"]
        del logargs["flair_dev"]
        del logargs["flair_test"]
        del logargs["flair_train"]
        del logargs["label_smoothing"]
        del logargs["max_sentences"]
        del logargs["rnn_cell_dim"]
        del logargs["test_data"]
        del logargs["threads"]
        del logargs["train_data"]
        del logargs["we_dim"]
        del logargs["word_dropout"]
        logargs["bert_embeddings"] = 1 if args.bert_embeddings_train else 0
        logargs["flair_embeddings"] = 1 if args.flair_train else 0
        logargs["elmo_embeddings"] = 1 if args.elmo_train else 0

        args.logdir = "logs/{}-{}-{}".format(
            os.path.basename(__file__),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("^.*/", "", value) if type(value) == str else value)
                      for key, value in sorted(logargs.items())))
        )
        if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself
        if not os.path.exists(args.logdir): os.mkdir(args.logdir)
        
        # Dump passed options to allow future prediction.
        with open("{}/options.json".format(args.logdir), mode="w") as options_file:
            json.dump(vars(args), options_file, sort_keys=True)

    # Postprocess args
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]
   
    # Load the data
    seq2seq = args.decoding == "seq2seq"
    train = morpho_dataset.MorphoDataset(args.train_data, max_sentences=args.max_sentences, seq2seq=seq2seq, bert_embeddings_filename=args.bert_embeddings_train, flair_filename=args.flair_train, elmo_filename=args.elmo_train)
    if args.dev_data:
        dev = morpho_dataset.MorphoDataset(args.dev_data, train=train, shuffle_batches=False, seq2seq=seq2seq, bert_embeddings_filename=args.bert_embeddings_dev, flair_filename=args.flair_dev, elmo_filename=args.elmo_dev)
    test = morpho_dataset.MorphoDataset(args.test_data, train=train, shuffle_batches=False, seq2seq=seq2seq, bert_embeddings_filename=args.bert_embeddings_test,flair_filename=args.flair_test, elmo_filename=args.elmo_test)

    # Load pretrained form embeddings
    if args.form_wes_model:
        args.form_wes_model = word2vec.load(args.form_wes_model)
    if args.lemma_wes_model:
        args.lemma_wes_model = word2vec.load(args.lemma_wes_model)

    # Load fasttext subwords embeddings
    if args.fasttext_model:
        args.fasttext_model = fasttext.load_model(args.fasttext_model)

    # Character-level embeddings
    args.including_charseqs = (args.cle_dim > 0)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args,
                      num_forms=len(train.factors[train.FORMS].words),
                      num_form_chars=len(train.factors[train.FORMS].alphabet),
                      num_lemmas=len(train.factors[train.LEMMAS].words),
                      num_lemma_chars=len(train.factors[train.LEMMAS].alphabet),
                      num_pos=len(train.factors[train.POS].words),
                      pretrained_form_we_dim=args.form_wes_model.vectors.shape[1] if args.form_wes_model else 0,
                      pretrained_lemma_we_dim=args.lemma_wes_model.vectors.shape[1] if args.lemma_wes_model else 0,
                      pretrained_fasttext_dim=args.fasttext_model.get_dimension() if args.fasttext_model else 0,
                      num_tags=len(train.factors[train.TAGS].words),
                      tag_bos=train.factors[train.TAGS].words_map["<bos>"],
                      tag_eow=train.factors[train.TAGS].words_map["<eow>"],
                      pretrained_bert_dim=train.bert_embeddings_dim(),
                      pretrained_flair_dim=train.flair_embeddings_dim(),
                      pretrained_elmo_dim=train.elmo_embeddings_dim(),
                      predict_only=args.predict)

    if args.predict:
        network.saver.restore(network.session, "{}/model".format(args.predict.rstrip("/")))
        print("Predicting test data", file=sys.stderr)
        network.predict("test", test, args, sys.stdout, evaluating=False)
    else:
        # Train
        for epochs, learning_rate in args.epochs:
            for epoch in range(epochs):
                network.train_epoch(train, learning_rate, args)
                dev_score = 0
                if args.dev_data:
                    dev_score = network.evaluate("dev", dev, args)
                    print("{}".format(dev_score))
        # Save network
        network.saver.save(network.session, "{}/model".format(args.logdir), write_meta_graph=False)
        # Test
        test_score = network.evaluate("test", test, args)
        print("{}".format(test_score))
