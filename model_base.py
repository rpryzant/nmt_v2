import abc
import time
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import input_pipeline

# TODO lots of needlessly shared stuff bt eval/train and inference, seperate out or clean up
class Model(namedtuple("TrainModel", ('graph', 'model', 'iterator', 'src_file', 'tgt_file', 'src_placeholder', 'batch_size_placeholder'))):
    pass


def build_model_graph(model_creator, config, mode="train"):
    assert mode in ["train", "eval"]

    if mode == "train":
        src_file = os.path.join(config.data_dir, "%s.%s" % (config.train_prefix, config.src))
        tgt_file = os.path.join(config.data_dir, "%s.%s" % (config.train_prefix, config.tgt))
    elif mode == "eval":
        src_file = os.path.join(config.data_dir, "%s.%s" % (config.eval_prefix, config.src))
        tgt_file = os.path.join(config.data_dir, "%s.%s" % (config.eval_prefix, config.tgt))

    src_vocab_file = os.path.join(config.data_dir, "%s.%s" % (config.vocab_prefix, config.src))
    tgt_vocab_file = os.path.join(config.data_dir, "%s.%s" % (config.vocab_prefix, config.tgt))

    graph = tf.Graph()

    with graph.as_default():
        iterator, tgt_vocab_table, reverse_tgt_vocab_table = \
            input_pipeline.get_iterator(
                src_file, tgt_file, src_vocab_file, tgt_vocab_file, config)

        model = model_creator(config, iterator, mode, tgt_vocab_table, reverse_tgt_vocab_table)

        return Model(graph=graph, model=model, iterator=iterator, src_file=src_file, tgt_file=tgt_file, src_placeholder=None, batch_size_placeholder=None)




def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print "  loaded %s model parameters from %s, time %.2fs" % \
        (name, ckpt, time.time() - start_time)
    return model



def create_or_load_model(model, model_dir, session, name):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)

    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print "  created %s model with fresh parameters, time %.2fs" % \
                        (name, time.time() - start_time)

    global_step = model.global_step.eval(session=session)
    return model, global_step        



class BaseModel(object):
    """ sequence to sequence base class
    """
    def __init__(self,
                 config, 
                 iterator, 
                 mode,
                 tgt_vocab_table,
                 reverse_tgt_vocab_table=None):
        assert mode in ['train', 'eval', 'test']

        if mode == "test":
            assert reverse_tgt_vocab_table is not None

        self.iterator = iterator
        # pull out batch size dynamically
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        self.config = config
        self.tgt_vocab_table = tgt_vocab_table

        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        self.num_layers = config.num_layers

        self.mode = mode

        # initializer for all matrices
        initializer = tf.random_uniform_initializer(
            -config.init_weight, config.init_weight, seed=config.random_seed)
        tf.get_variable_scope().set_initializer(initializer)

        # init global step
        self.global_step = tf.Variable(0, trainable=False)
        # make embeddings
        self.encoder_embeddings, self.decoder_embeddings = self.make_embeddings()
        # make graph
        self.loss, self.logits, self.sample_ids = self.build_graph()

        if self.mode == 'test':
            self.preds = reverse_tgt_vocab_table.lookup(
                tf.to_int64(self.sample_ids))
        else:
            self.preds = tf.argmax(self.logits, axis=2)
            self.train_op = self._make_training_op()
            self.word_count = tf.reduce_sum(
                self.iterator.target_sequence_length) + tf.reduce_sum(
                self.iterator.source_sequence_length)

        # remember inputs + outputs
        self.step_src = self.iterator.source
        self.step_src_len = self.iterator.source_sequence_length
        self.step_tgt = self.iterator.target_output
        self.step_tgt_len = self.iterator.target_sequence_length

        # tf boilerplate stats
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())


    def make_embeddings(self):
        with tf.variable_scope("embeddings"):
            if self.config.share_vocab:
                embedding = tf.get_variable(
                    "embedding_share", [self.src_vocab_size, self.config.src_embed_size])
                encoder_embeddings = embedding
                decoder_embeddings = embedding
            else:
                with tf.variable_scope("encoder"):
                    encoder_embeddings = tf.get_variable(
                        "embedding_encoder",
                        [self.src_vocab_size, self.config.src_embed_size])
                with tf.variable_scope("decoder"):
                    decoder_embeddings = tf.get_variable(
                        "embedding_decoder",
                        [self.tgt_vocab_size, self.config.tgt_embed_size])

        return encoder_embeddings, decoder_embeddings


    def build_graph(self):
        with tf.variable_scope("encoder"):
            encoder_outputs, encoder_state = self._build_encoder()
        with tf.variable_scope("decoder"):
            logits, sample_ids = self._build_decoder(encoder_outputs, encoder_state)

        if self.mode != "test":
            with tf.variable_scope("loss"):
                loss = self._compute_loss(logits)
        else:
            loss = tf.no_op()

        return loss, logits, sample_ids

    @abc.abstractmethod
    def _build_encoder(self):
        pass


    @abc.abstractmethod
    def _build_decoder_cell(self, encoder_outputs, encoder_state):
        pass


    def _build_decoder(self, encoder_outputs, encoder_state):
        output_layer = layers_core.Dense(
            self.config.tgt_vocab_size, use_bias=False, name="out_projection")

        cell, initial_state = self._build_decoder_cell(
            encoder_outputs, encoder_state)

        # train or eval (argmax)
        if self.mode != 'test':
            target_input = self.iterator.target_input
            target_embeddings = tf.nn.embedding_lookup(
                self.decoder_embeddings, target_input)
            # argmax sampler
            sampler = tf.contrib.seq2seq.TrainingHelper(
                target_embeddings, self.iterator.target_sequence_length)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell, sampler, initial_state)
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, swap_memory=True)  # move tensors to cpu after computation, avoid memory issues on long seqs

            # applying projection all at once is faster than 
            #    the per-timestep behavior in test
            # apply a new scope here because tf.Decoder applies output_layer
            #    within this scope by default
            with tf.variable_scope("decoder"):
                logits = output_layer(outputs.rnn_output)
            sample_ids = outputs.sample_id

        # test (beam search)
        else:
            tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(self.config.sos)),
                                 tf.int32)
            tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(self.config.eos)),
                                 tf.int32)
            beam_width = self.config.beam_width
            length_penalty_weight = self.config.length_penalty_weight
            start_tokens = tf.fill([self.batch_size], tgt_sos_id)
            end_token = tgt_eos_id

            # max decoding steps
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(self.iterator.source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))

            if beam_width > 0:
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cell,
                    embedding=self.decoder_embeddings,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=initial_state,
                    beam_width=beam_width,
                    output_layer=output_layer,
                    length_penalty_weight=length_penalty_weight)
            else:
                # test argmax sampler
                sampler = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding_decoder, start_tokens, end_token)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, sampler, initial_state, output_layer=output_layer)

            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=maximum_iterations, swap_memory=True)

            if beam_width > 0:
                logits = tf.no_op()
                sample_ids = outputs.predicted_ids
            else:
                logits = outputs.rnn_output  # already projected
                sample_ids = outputs.sample_id

        return logits, sample_ids


    def _compute_loss(self, logits):
        targets = self.iterator.target_output
        seq_lens = self.iterator.target_sequence_length
        time_axis = 1
        max_time = targets.shape[time_axis].value or tf.shape(targets)[time_axis]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)
        mask = tf.sequence_mask(seq_lens, max_time, dtype=logits.dtype)
        loss = tf.reduce_sum(crossent * mask) / tf.to_float(tf.reduce_sum(seq_lens))
        return loss

    def _make_training_op(self):
        if self.config.optimizer == 'sgd':
            self.learning_rate = tf.cond(
                self.global_step < self.config.start_decay_step,
                lambda: tf.constant(self.config.learning_rate),
                lambda: tf.train.exponential_decay(
                    self.config.learning_rate,
                    (self.global_step - self.config.start_decay_step),
                    self.config.decay_steps,
                    self.config.decay_factor,
                    staircase=True),
                name='learning_rate')
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adam':
            assert self.config.learning_rate < 0.007
            self.learning_rate = tf.constant(self.config.learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(
            gradients, self.config.max_gradient_norm)

        tf.summary.scalar("grad_norm", gradient_norm)
        tf.summary.scalar("clipped_norm", tf.global_norm(clipped_gradients))
        tf.summary.scalar("learning_rate", self.learning_rate)

        train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        return train_op


    def train(self, sess, debug=False):
        assert self.mode == "train"
        ops = [self.train_op,
               self.loss,
               self.word_count,
               self.global_step,
               self.summaries]
        if debug:
            ops += [self.step_src, self.step_src_len, self.step_tgt, self.step_tgt_len, self.preds]

        return sess.run(ops)


    def eval(self, sess):
        assert self.mode == "eval"
        return sess.run([self.loss, self.word_count])


    def test(self, sess):
        assert self.mode == "test"
        return sess.run([self.logits, self.preds])


    ###########################
    #  utility functions for subclasses
    ###########################
    def _build_rnn_cell(self, layers=None):
        def _single_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(
                self.config.num_units, forget_bias=self.config.forget_bias)

            dropout = self.config.dropout if self.mode == "train" else 0.0
            cell = tf.contrib.rnn.DropoutWrapper(
                cell = cell, input_keep_prob=(1.0 - dropout))

            return cell

        cell_list = [_single_cell() for _ in range((layers or self.config.num_layers))]
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        return cell


    def build_unidirectional_encoder(self, source_embedded):
        cell = self._build_rnn_cell()
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            source_embedded,
            dtype=tf.float32,
            sequence_length=self.iterator.source_sequence_length)
        return encoder_outputs, encoder_state


    def build_bidirectional_encoder(self, source_embedded):
        fw_cell = self._build_rnn_cell(layers=self.config.num_layers / 2)
        bw_cell = self._build_rnn_cell(layers=self.config.num_layers / 2)
        bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            source_embedded,
            dtype=tf.float32,
            sequence_length=self.iterator.source_sequence_length)
        return tf.concat(bi_output, -1), bi_state



















