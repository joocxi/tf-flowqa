import os
import tensorflow as tf

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


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        pass
    elif config.mode == "preprocess":
        pass
    elif config.mode == "debug":
        pass
    elif config.mode == "test":
        pass


if __name__ == "__main__":
    tf.app.run()
