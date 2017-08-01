import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

UNK_ID = 0

def check_vocab(vocab_file, out_dir, sos, eos, unk):
    """ verify vocab_file is proper """
    assert os.path.exists(vocab_file), "The vocab file %s does not exist" % vocab_file

    lines = map(lambda x: x.strip(), open(vocab_file).readlines())

    assert lines[0] == unk and lines[1] == sos and lines[2] == eos, \
        "The first words in %s are not %s, %s, %s" % (vocab_file, unk, sos, eos)

    return vocab_file, len(lines)


def create_vocab_tables(src_vocab_file, tgt_vocab_file, config):
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)
    if config.share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
            tgt_vocab_file, default_value=UNK_ID)

    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=config.unk)

    return src_vocab_table, tgt_vocab_table, reverse_tgt_vocab_table

