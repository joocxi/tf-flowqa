import os
import tensorflow as tf

from preprocess import prepro
from trainer import train

flags = tf.flags

data_dir = "data"
record_dir = "record"

train_file = os.path.join(data_dir, "coqa-train-v1.0.json")
dev_file = os.path.join(data_dir, "coqa-dev-v1.0.json")

train_record_file = os.path.join(record_dir, "train.tfrecords")
dev_record_file = os.path.join(record_dir, "dev.tfrecords")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)

flags.DEFINE_string("mode", "train", "train/debug/test")

flags.DEFINE_string("data_dir", data_dir, "")
flags.DEFINE_string("record_dir", record_dir, "")

flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("dev_file", dev_file, "")

flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")
flags.DEFINE_integer("hidden_dim", 75, "size of hidden dim")
flags.DEFINE_integer("grad_clip", 5, "global norm gradient clipping")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "preprocess":
        prepro(config)
    elif config.mode == "debug":
        #TODO: set debug configuration
        train(config)
    elif config.mode == "evaluate":
        pass


if __name__ == "__main__":
    tf.app.run()
