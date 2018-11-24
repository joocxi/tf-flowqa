import os
import tensorflow as tf

from preprocess import prepro
from trainer import train

flags = tf.flags

data_dir = "data"
log_dir = "log"
train_file = os.path.join(data_dir, "coqa-train-v1.0.json")
dev_file = os.path.join(data_dir, "coqa-dev-v1.0.json")
glove_word_file = os.path.join(data_dir, "glove", "glove.840B.300d.txt")
cove_word_file = os.path.join(data_dir, "cove", "Keras_CoVe.h5")
elmo_options_file = os.path.join(data_dir, "elmo", "options.json")
elmo_weight_file = os.path.join(data_dir, "elmo", "lm_weights.hdf5")

record_dir = "record"
train_record_file = os.path.join(record_dir, "train.tfrecords")
dev_record_file = os.path.join(record_dir, "dev.tfrecords")

prepro_dir = "processed"
elmo_vocab_file = os.path.join(prepro_dir, "elmo_vocab.txt")
glove_word_emb_file = os.path.join(prepro_dir, "glove_word_emb.json")
glove_word2idx_file = os.path.join(prepro_dir, "glove_word2idx.json")


if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
if not os.path.exists(prepro_dir):
    os.makedirs(prepro_dir)

flags.DEFINE_string("mode", "train", "train/debug/test")

flags.DEFINE_string("log_dir", log_dir, "")
flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("dev_file", dev_file, "")
flags.DEFINE_string("glove_word_file", glove_word_file, "")
flags.DEFINE_string("glove_word2idx_file", glove_word2idx_file, "")
flags.DEFINE_string("cove_word_file", cove_word_file, "")
flags.DEFINE_string("elmo_options_file", elmo_options_file, "")
flags.DEFINE_string("elmo_weight_file", elmo_weight_file, "")

flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")

flags.DEFINE_string("elmo_vocab_file", elmo_vocab_file, "")
flags.DEFINE_string("glove_word_emb_file", glove_word_emb_file, "")
flags.DEFINE_integer("glove_word_size", int(2000), "")
# set global config
flags.DEFINE_integer("capacity", 1000, "capacity of buffer")
flags.DEFINE_integer("max_char_length", 50, "")
flags.DEFINE_integer("turn_limit", 40, "")
flags.DEFINE_integer("para_limit", 400, "")
flags.DEFINE_integer("ques_limit", 60, "")
flags.DEFINE_integer("glove_dim", 300, "")
flags.DEFINE_integer("cove_dim", 600, "")
flags.DEFINE_integer("elmo_dim", 1024, "")
flags.DEFINE_integer("hidden_dim", 75, "size of hidden dim")
flags.DEFINE_integer("attention_dim", 75, "size of hidden dim")

# set training config
flags.DEFINE_integer("batch_size", 1, "batch of training data")
flags.DEFINE_integer("grad_clip", 5, "global norm gradient clipping")
flags.DEFINE_integer("dev_batch_size", 4, "batch of validation data")
flags.DEFINE_integer("dev_steps", 113, "number of validation steps")
flags.DEFINE_float("learning_rate", 10e-3, "learning rate")
flags.DEFINE_integer("train_steps", 100000, "number of training steps")
flags.DEFINE_integer("dev_period", 5000, "validation period")
flags.DEFINE_integer("save_period", 500, "save period")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "preprocess":
        prepro(config)
    elif config.mode == "debug":
        config.train_steps = 2
        config.dev_steps = 1
        config.dev_period = 1
        config.save_period = 1
        train(config)
    elif config.mode == "evaluate":
        pass


if __name__ == "__main__":
    tf.app.run()
