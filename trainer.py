import tensorflow as tf
import ujson as json
import numpy as np

from model import FlowQA
from bilm import Batcher, BidirectionalLanguageModel


def train(config):

    with open(config.word_emb_file, "r") as wm:
        word_mat = np.array(json.load(wm), dtype=np.float32)

    # TODO: create train/val iterator
    train_dataset = None
    dev_dataset = None
    handle = None
    iterator = None
    train_iterator = None
    dev_iterator = None

    # TODO: init model
    model = FlowQA(config=config, iterator=None, word_mat=word_mat)
    pass

    # TODO: init session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        # TODO: implement training
        pass
