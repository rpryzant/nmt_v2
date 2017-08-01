import yaml
from collections import namedtuple
import os

import model_base
import models
import dummies
import vocab_utils as vocab_utils
import train

def load_config(filename):
    d = yaml.load(open(filename).read())
    c = namedtuple("config", d.keys())(**d)

    src_vocab_file = os.path.join(c.data_dir, c.vocab_prefix + "." + c.src)
    src_vocab_file, src_vocab_size = vocab_utils.check_vocab(
        src_vocab_file, c.data_dir, c.sos, c.eos, c.unk)
    c = c._replace(src_vocab_size=src_vocab_size)

    if not c.share_vocab:
        tgt_vocab_file = os.path.join(c.data_dir, c.vocab_prefix + "." + c.tgt)
        tgt_vocab_file, tgt_vocab_size = vocab_utils.check_vocab(
            tgt_vocab_file, c.data_dir, c.sos, c.eos, c.unk)
        c = c._replace(tgt_vocab_size=tgt_vocab_size)

    if not os.path.exists(c.out_dir):
        os.makedirs(c.out_dir)

    return c

c = load_config("config.yaml")

train.train(c)




