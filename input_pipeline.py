from collections import namedtuple

import tensorflow as tf

import utils.vocab_utils as vocab_utils


class BatchedInput(namedtuple("BatchedInput",
                                            ("initializer",
                                             "source",
                                             "target_input",
                                             "target_output",
                                             "source_sequence_length",
                                             "target_sequence_length"))):
    pass





def get_test_iterator(src_dataset, src_vocab_table, batch_size, config):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(config.eos)), tf.int32)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    src_dataset = src_dataset.map(lambda src: src[:config.src_max_len])

    src_dataset = src_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    if config.reverse_src:
        src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))

    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
        return x.padded_batch(
            config.batch_size,
            padded_shapes=(tf.TensorShape([None]),
                           tf.TensorShape([])),
            padding_values=(src_eos_id,
                            0))

    batched_dataset = batching_func(src_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    src_ids, src_seq_len = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None)



def get_iterator(src_file, tgt_file, src_vocab_file, tgt_vocab_file, config, threads=4):
    output_buffer_size = config.batch_size * 1000

    src_dataset = tf.contrib.data.TextLineDataset(src_file)
    tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file)
    src_vocab_table, tgt_vocab_table, reverse_tgt_vocab_table = \
        vocab_utils.create_vocab_tables(
            src_vocab_file, tgt_vocab_file, config)

    src_eos_id = tf.cast(
        src_vocab_table.lookup(tf.constant(config.eos)),
        tf.int32)

    tgt_sos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(config.sos)),
        tf.int32)
    tgt_eos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(config.eos)),
        tf.int32)

    # pair up src + tgt sentences
    src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

    # shuffle (not sure what the buffer is doing...)
    src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, config.random_seed)

    # break sentences into words
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_threads=threads,
        output_buffer_size=output_buffer_size)

    # make sure 0 < len(seq) < max_len
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:config.src_max_len], tgt[:config.tgt_max_len]),
        num_threads=threads,
        output_buffer_size=output_buffer_size)

    # reverse source if the user asked for it
    if config.reverse_src:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.reverse(src, axis=[0]), tgt),
            num_threads=threads,
            output_buffer_size=output_buffer_size)

    # convert word strings to ids 
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_threads=threads, output_buffer_size=output_buffer_size)

    # wrap tgt examples with sos and eos
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([tgt_sos_id], tgt), 0),
                          tf.concat((tgt, [tgt_eos_id]), 0)),
        num_threads=threads, output_buffer_size=output_buffer_size)

    # add in word counts. subtract one from target to avoid counting sos/eos
    #   TODO -- TROUBLESHOOT
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_out)),
        num_threads=threads, output_buffer_size=output_buffer_size)

    # batch up
    def batch_pad(dataset):
        return dataset.padded_batch(
            config.batch_size,
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([None]),  # tgt_output
                           tf.TensorShape([]),      # src_len
                           tf.TensorShape([])),     # tgt_len
            padding_values=(src_eos_id,
                            tgt_eos_id,
                            tgt_eos_id,
                            0,   # unused
                            0))  # unused

    # bucket up
    if config.num_buckets > 1:
        # maps examples to keys (buckets)
        def key_func(src, tgt_in, tgt_out, src_len, tgt_len):
            bucket_width = (config.src_max_len + config.num_buckets - 1) // config.num_buckets
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(config.num_buckets, bucket_id))
        # batches and pads the examples for a bucket
        def reduce_func(unused, windowed_data):
            return batch_pad(windowed_data)

        batched_dataset = src_tgt_dataset.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=config.batch_size)
    else:
        batched_dataset = batch_pad(src_tgt_dataset)

    # create an iterator from this dataset
    batched_iter = batched_dataset.make_initializable_iterator()
    # pull out some values to use as "placeholder"-type things
    src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len = \
        batched_iter.get_next()
    # return a NamedTuple of this stuff (as well as the initializer )
    return BatchedInput(
            initializer=batched_iter.initializer,
            source=src_ids,
            target_input=tgt_input_ids,
            target_output=tgt_output_ids,
            source_sequence_length=src_seq_len,
            target_sequence_length=tgt_seq_len), \
        tgt_vocab_table, reverse_tgt_vocab_table














