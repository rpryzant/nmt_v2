import codecs
import os

import tensorflow as tf

import model_base
import utils
import input_pipeline




def translate_file(test_model, test_sess, out_file, eos, src_file):
    # TODO take out dir too
    with test_model.graph.as_default():
        loaded_test_model, global_step = model_base.create_or_load_model(
            test_model.model, out_dir, test_sess, "test")

    test_sess.run(test_model.iterator.initializer,
        feed_dict={test_model.src_placeholder: src_file})

    print ' translating source'

    preds = None
    while True:
        try:
            _, nmt_outputs = loaded_test_model.test(test_sess)
            preds = nmt_outputs if preds is None else np.concatenate(preds, nmt_outputs, axis=0)
        except:
            break

    nmt_outputs = [
        inference.format_decoding(pred, target_beam=0, eos=eos) \
        for pred in preds]

    print ' writing translations to ', out_file

    with open(out_file, 'w') as f:
        f.write('\n'.join(nmt_outputs))



def load_data(filepath):
    with codecs.getreader("utf-8")(open(filepath)) as f:
        data = f.read().splitlines()
    return data


def build_inference_graph(model_creator, config):
    graph = tf.Graph()
    with graph.as_default():

        src_file = os.path.join(config.data_dir, "%s.%s" % (config.test_prefix, config.src))
        tgt_file = os.path.join(config.data_dir, "%s.%s" % (config.test_prefix, config.tgt))

        src_vocab_file = os.path.join(config.data_dir, "%s.%s" % (config.vocab_prefix, config.src))
        tgt_vocab_file = os.path.join(config.data_dir, "%s.%s" % (config.vocab_prefix, config.tgt))

        src_vocab_table, tgt_vocab_table, reverse_tgt_vocab_table = \
            utils.create_vocab_tables(
                src_vocab_file, tgt_vocab_file, config)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            src_placeholder)

        iterator = input_pipeline.get_test_iterator(
            src_dataset, src_vocab_table, batch_size_placeholder, config)

        model = model_creator(config, iterator, "test", tgt_vocab_table, reverse_tgt_vocab_table)

    return model_base.Model(graph=graph, model=model, iterator=iterator, src_file=src_file, tgt_file=tgt_file, src_placeholder=src_placeholder, batch_size_placeholder=batch_size_placeholder)


def format_decoding(outputs, target_beam=None, eos=None):
    output = list(outputs[:,target_beam])  # if np.ndarray

    if eos and eos in output:
        output = output[:output.index(eos)]

    output = " ".join(output)

    return output



