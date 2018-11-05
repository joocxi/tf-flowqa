import tensorflow as tf
from layer import rnn, integration_flow, question_attention, fully_aware_attention


class FlowQA(object):

    def __init__(self, config, iterator, word_mat=None, trainable=True):
        self.config = config
        self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)

        self.context_idxs, self.questions_idxs, self.starts, self.ends, self.em = iterator.get_next()
        """
        context_idxs: (batch_size, para_limit)
        questions_idxs: (batch_size, turn_limit, ques_limit)
        starts: (batch_size, turn_limit, para_limit)
        ends: (batch_size, turn_limit, para_limit)
        em: (batch_size, turn_limit, para_limit)
        """

        batch_size = config.batch_size

        # TODO: create mask
        # self.context_mask = tf.cast(self.context_idxs, tf.bool)
        # self.questions_mask = tf.cast(self.questions_idxs, tf.bool)

        # context_length = tf.reduce_sum(tf.cast(self.context_mask, tf.int32), axis=1)
        # max_context_length = tf.reduce_max(context_length)
        # self.context_idxs = tf.slice(self.context_idxs, [0, 0], [batch_size, max_context_length])

        self.ready()

        if trainable:
            self.learning_rate = tf.get_variable("learning_rate", shape=[], dtype=tf.float32, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.optimizer.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        with tf.variable_scope("embedding"):
            # shape: (batch_size, para_limit, embedding_dim)
            context_emb = tf.nn.embedding_lookup(self.word_mat, self.context_idxs)
            # shape: (batch_size, turn_limit, ques_limit, embedding_dim)
            questions_emb = tf.nn.embedding_lookup(self.word_mat, self.questions_idxs)
            
        with tf.variable_scope("encoding"):
            # context encoding - attention on question
            # shape: (batch_size, turn_limit, para_limit, embedding_dim)
            g = question_attention(context_emb, questions_emb)
            em = tf.expand_dims(self.em, axis=-1)
            c = tf.tile(tf.expand_dims(context_emb, axis=1), [1, turn_limit, 1, 1])

            # shape: (batch_size, turn_limit, para_limit, embedding_dim + 1 + embedding_dim = context_dim)
            c_0 = tf.concat([c, em, g], -1)

            # question encoding - question integration
            input_size = None
            ques_limit = None

            # shape: (batch_size*turn_limit, ques_limit, embedding_dim)
            _q = tf.reshape(questions_emb, [batch_size*turn_limit, ques_limit, embedding_dim])

            bi_lstm = rnn(num_layers=2, bidirectional=True, num_units=config.hidden_dim,
                        batch_size=batch_size*turn_limit, input_size=input_size, is_train=self.is_train)
            # shape: (batch_size*turn_limit, ques_limit, 4*hidden_dim)
            _q_12 = bi_lstm(_q, seq_len=ques_limit, concat_layers=True)
            # shape: (batch_size, turn_limit, ques_limit, 4*hidden_dim)
            q_12 = tf.reshape(_q_12, [batch_size, turn_limit, ques_limit, 4*hidden_dim])
            # shape: (batch_size, turn_limit, ques_limit, 2*hidden_dim)
            q_1 = tf.slice(q_12, [0, 0, 0, 0], [-1, -1, -1, 2*hidden_dim])
            q_2 = tf.slice(q_12, [0, 0, 0, 2*hidden_dim], [-1, -1, -1, -1])

            # shape: (batch_size, turn_limit, ques_limit)
            q_2_fc = tf.squeeze(tf.layers.dense(q_2, 1, use_bias=False), axis=-1)
            q_2_logits = tf.nn.softmax(q_2_fc)

            # shape: (batch_size, turn_limit, 2*hidden_dim)
            q_tilde = tf.squeeze(tf.matmul(tf.expand_dims(q_2_logits, axis=2), q_2), axis=2)

            uni_lstm = rnn(num_layers=1, bidirectional=False, num_units=config.hidden_dim,
                        batch_size=batch_size, input_size=hidden_dim, is_train=self.is_train)
            # shape: (batch_size, turn_limit, hidden_dim)
            p = uni_lstm(q_tilde, seq_len=turn_limit, concat_layers=False)

        with tf.variable_scope("reasoning"):
            # integration-flow x2
            # shape: (batch_size, turn_limit, para_limit, embedding_dim)
            c_1 = integration_flow(c_0)
            c_2 = integration_flow(c_1)

            # attention on question
            # shape: (batch_size, turn_limit, para_limit, concat_dim)
            c_concat = tf.concat([c_0, c_1, c_2], axis=-1)
            # shape: (batch_size, turn_limit, ques_limit, concat_dim)
            q_concat = tf.concat([question_emb, q_1, q_2], axis=-1)

            # shape: (batch_size, turn_limit, para_limit, hidden_dim)
            q_hat = fully_aware_attention(c_concat, q_concat, q_2)

            # integration-flow
            c_q_concat = tf.concat([c_2, q_hat], axis=-1)
            c_3 = integration_flow(c_q_concat)

            # attention on context
            c_concat = tf.concat([c_1, c_2, c_3], axis=-1)
            # shape: (batch_size, turn_limit, para_limit, hidden_dim)
            c_hat = fully_aware_attention(c_concat, c_concat, c_3)

            # integration
            cc_concat = tf.concat([c_3, c_hat], axis=-1)
            bi_lstm = rnn(num_layers=1, bidirectional=True, num_units=config.hidden_dim,
                        batch_size=batch_size, input_size=hidden_dim, is_train=self.is_train)
            _cc_concat = tf.reshape(cc_concat, [batch_size*turn_limit, para_limit, embedding_dim])
            _c_4 = bi_lstm(_cc_concat, seq_len=ques_limit, concat_layers=False)
            c_4 = tf.reshape(_c_4, [batch_size, turn_limit, para_limit, embedding_dim])

        with tf.variable_scope("prediction"):
            # shape: (batch_size, turn_limit, embedding_dim)
            p_fc = tf.layers.dense(p, embedding_dim, use_bias=False)

            # shape: (batch_size, turn_limit, para_limit)
            start_logits = tf.squeeze(tf.matmul(c_4, tf.expand_dims(p_fc, axis=-1)), axis=-1)
            start_probs = tf.nn.softmax(start_logits)

            # shape: (batch_size, turn_limit, embedding_dim)
            c_4_avg = tf.squeeze(tf.matmul(tf.expand_dims(start_probs, axis=2), c_4), axis=2)
            gru = tf.contrib.rnn.GruCell(hidden_dim)
            # shape: (batch_size, turn_limit, embedding_dim)
            p_hat = gru(p, c_4_avg)
            # shape: (batch_size, turn_limit, embedding_dim)
            p_hat_fc = tf.layers.dense(p_hat, embedding_dim, use_bias=False)
            # shape: (batch_size, turn_limit, para_limit)
            end_logits = tf.squeeze(tf.matmul(c_4, tf.expand_dims(p_hat_fc, axis=-1)), axis=-1)
            end_probs = tf.nn.softmax(end_logits)

            # TODO: no-answer
            # shape: (batch_size, turn_limit, embedding_dim)
            c_4_sum = tf.reduce_sum(c_4, axis=2)
            c_4_max = tf.reduce_max(c_4, axis=2)
            # shape: (batch_size, turn_limit, concat_dim)
            c_4_concat = tf.concat([c_4_sum, c_4_max], axis=-1)
            # shape: (batch_size, turn_limit, concat_dim)
            p_fc = tf.layers.dense(p, concat_dim, use_bias=False)
            # shape: (batch_size, turn_limit)
            no_logits = tf.squeeze(tf.matmul(tf.expand_dims(c_4_concat, axis=2), tf.expand_dims(p_fc, axis=-1)), axis=-1)
            no_probs = tf.nn.sigmoid(no_logits)

        with tf.name_scope("loss"):
            # TODO:
            self.loss = None
