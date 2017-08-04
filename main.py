import yaml
from collections import namedtuple
import os
import argparse

import model_base
import models
import dummies
import utils
import train




def process_command_line():
    """
    Return a 1-tuple: (args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """

    parser = argparse.ArgumentParser(description='usage') # add description
    parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='gpu')
    args = parser.parse_args()
    return args




def load_config(filename):
    d = yaml.load(open(filename).read())
    c = namedtuple("config", d.keys())(**d)

    src_vocab_file = os.path.join(c.data_dir, c.vocab_prefix + "." + c.src)
    src_vocab_file, src_vocab_size = utils.check_vocab(
        src_vocab_file, c.data_dir, c.sos, c.eos, c.unk)
    c = c._replace(src_vocab_size=src_vocab_size)

    if c.share_vocab:
        c = c._replace(tgt_vocab_size=src_vocab_size)
    else:
        tgt_vocab_file = os.path.join(c.data_dir, c.vocab_prefix + "." + c.tgt)
        tgt_vocab_file, tgt_vocab_size = utils.check_vocab(
            tgt_vocab_file, c.data_dir, c.sos, c.eos, c.unk)
        c = c._replace(tgt_vocab_size=tgt_vocab_size)

    if not os.path.exists(c.out_dir):
        os.makedirs(c.out_dir)

    return c


args = process_command_line()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
c = load_config("config.yaml")
train.train(c)




