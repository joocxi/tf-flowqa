import tensorflow as tf


def get_parser(config):
    def parse(example):
        turn_limit = config.turn_limit
        para_limit = config.para_limit
        ques_limit = config.ques_limit
        max_char_length = config.max_char_length
        
        features = tf.parse_single_example(example,
                                features={
                                    "context_idxs": tf.FixedLenFeature([], tf.string),
                                    "questions_idxs": tf.FixedLenFeature([], tf.string),
                                    "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                    "questions_char_idxs": tf.FixedLenFeature([], tf.string),
                                    "starts": tf.FixedLenFeature([], tf.string),
                                    "ends": tf.FixedLenFeature([], tf.string),
                                    "em": tf.FixedLenFeature([], tf.string),
                                    "yes_answers": tf.FixedLenFeature([], tf.string),
                                    "no_answers": tf.FixedLenFeature([], tf.string),
                                    "unk_answers": tf.FixedLenFeature([], tf.string),
                                    "span_flag": tf.FixedLenFeature([], tf.string)
                                })

        context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int32), [para_limit])
        questions_idxs = tf.reshape(tf.decode_raw(features["questions_idxs"], tf.int32), [turn_limit, ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit + 2, max_char_length])
        questions_char_idxs = tf.reshape(tf.decode_raw(features["questions_char_idxs"], tf.int32), [turn_limit, ques_limit + 2, max_char_length])
        starts = tf.reshape(tf.decode_raw(features["starts"], tf.float32), [turn_limit, para_limit])
        ends = tf.reshape(tf.decode_raw(features["ends"], tf.float32), [turn_limit, para_limit])
        em = tf.reshape(tf.decode_raw(features["em"], tf.int32), [turn_limit, para_limit])
        yes_answers = tf.reshape(tf.decode_raw(features["yes_answers"], tf.int32), [turn_limit])
        no_answers = tf.reshape(tf.decode_raw(features["no_answers"], tf.int32), [turn_limit])
        unk_answers = tf.reshape(tf.decode_raw(features["unk_answers"], tf.int32), [turn_limit])
        span_flag = tf.reshape(tf.decode_raw(features["span_flag"], tf.int32), [turn_limit])
        return context_idxs, questions_idxs, context_char_idxs, questions_char_idxs, \
            starts, ends, em, yes_answers, no_answers, unk_answers, span_flag
    return parse    


def get_train_dataset(record_file, parser, config):
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser).shuffle(config.capacity).repeat()
    dataset = dataset.batch(config.batch_size)
    return dataset


def get_dev_dataset(record_file, parser, config):
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser).shuffle(config.capacity).repeat()
    dataset = dataset.batch(config.batch_size)
    return dataset
