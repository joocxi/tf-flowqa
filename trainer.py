import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm

from model import FlowQA
from bilm import Batcher, BidirectionalLanguageModel
from util import get_parser, get_train_dataset, get_dev_dataset


def train(config):

    with open(config.glove_word_emb_file, "r") as wm:
        word_mat = np.array(json.load(wm), dtype=np.float32)

    # create train/dev iterator
    parser = get_parser(config)
    train_dataset = get_train_dataset(config.train_record_file, parser, config)
    dev_dataset = get_dev_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    # init model
    model = FlowQA(config=config, iterator=iterator, word_mat=word_mat)

    # init session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:

        writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())

        sess.run(tf.assign(model.learning_rate, tf.constant(config.learning_rate, dtype=tf.float32)))
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        for _ in tqdm(range(config.train_steps)):
            global_step = sess.run(model.global_step) + 1
            loss, _ = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
            if global_step % config.save_period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss)])
                writer.add_summary(loss_sum, global_step)
            if global_step % config.dev_period == 0:
                sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
                dev_losses = []
                for _ in tqdm(range(config.dev_steps)):
                    dev_loss = sess.run(model.loss, feed_dict={handle: dev_handle})
                    dev_losses.append(dev_loss)

                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
                dev_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=np.mean(dev_loss))])
                writer.add_summary(dev_loss_sum, global_step)
                writer.flush()

