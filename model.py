import tensorflow as tf
from layer import rnn, integration_flow, question_attention, fully_aware_attention, softmax_mask
from bilm import BidirectionalLanguageModel, weight_layers
from keras.models import load_model


class FlowQA(object):

    def __init__(self, config, iterator, word_mat=None, trainable=True):
        self.config = config
        self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)

        self.word_mat = tf.get_variable("word_mat",
                                        initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)

        self.context_idxs, self.questions_idxs, self.context_char_idxs, self.questions_char_idxs, \
            self.starts, self.ends, self.em, self.yes_answers, self.no_answers, self.unk_answers, self.span_flag = iterator.get_next()

        self.batch_size = config.batch_size
        self.hidden_dim = config.hidden_dim

        # create mask
        self.context_mask = tf.cast(self.context_idxs, tf.bool)
        context_length = tf.reduce_sum(tf.cast(self.context_mask, tf.int32), axis=-1)
        self.para_size = tf.reduce_max(context_length)
        self.questions_mask = tf.cast(self.questions_idxs, tf.bool)
        # shape: (batch_size, turn_size)
        questions_length = tf.reduce_sum(tf.cast(self.context_mask, tf.int32), axis=-1)
        self.ques_size = tf.reduce_max(questions_length)
        self.turn_mask = tf.cast(questions_length, tf.bool)
        # shape: (batch_size)
        turns_length = tf.reduce_sum(tf.cast(turn_mask, tf.int32), axis=-1)
        self.turn_size = tf.reduce_max(turns_length)

        # slice to get the desired tensors
        self.context_idxs = tf.slice(self.context_idxs, [0, 0], [self.batch_size, self.para_size])
        self.context_mask = tf.slice(self.context_mask, [0, 0], [self.batch_size, self.para_size])
        self.questions_idxs = tf.slice(self.questions_idxs, [0, 0, 0], [self.batch_size, self.turn_size, self.ques_size])
        self.questions_mask = tf.slice(self.questions_mask, [0, 0, 0], [self.batch_size, self.turn_size, self.ques_size])
        self.context_char_idxs = tf.slice(self.context_char_idxs, [0, 0, 0], [self.batch_size, self.para_size + 2, -1])
        self.questions_char_idxs = tf.slice(questions_char_idxs, [0, 0, 0, 0], [self.batch_size, self.turn_size, self.ques_size + 2, -1])
        self.starts = tf.slice(self.starts, [0, 0, 0], [self.batch_size, self.turn_size, self.para_size])
        self.ends = tf.slice(self.ends, [0, 0, 0], [self.batch_size, self.turn_size, self.para_size])
        self.em = tf.slice(self.em, [0, 0, 0], [self.batch_size, self.turn_size, self.para_size])
        self.yes_answers = tf.slice(self.yes_answers, [0, 0], [self.batch_size, self.turn_size])
        self.no_answers = tf.slice(self.no_answers, [0, 0], [self.batch_size, self.turn_size])
        self.unk_answers = tf.slice(self.unk_answers, [0, 0], [self.batch_size, self.turn_size])
        self.turn_mask = tf.slice(self.turn_mask, [0, 0], [self.batch_size, self.turn_size])

        # construct model
        self.ready()

        if trainable:
            self.learning_rate = tf.get_variable("learning_rate", shape=[], dtype=tf.float32, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.optimizer.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        questions_batch = self.batch_size * self.turn_size
        with tf.variable_scope("embedding"):
            # shape: (batch_size, para_size, glove_dim)
            context_glove = tf.nn.embedding_lookup(self.word_mat, self.context_idxs)
            # shape: (batch_size, turn_size, ques_size, glove_dim)
            questions_glove = tf.nn.embedding_lookup(self.word_mat, self.questions_idxs)

            with tf.variable_scope("elmo"):
                # init bilm
                bilm = BidirectionalLanguageModel(self.config.elmo_options_file, self.config.elmo_weight_file)
                # shape: (batch_size, 3, para_size, 1024)
                context_lm = bilm(self.context_char_idxs)
                # shape: (batch_size*turn_size, ques_size, max_char_length)
                _questions_char_idxs = tf.reshape(self.questions_char_idxs, [questions_batch, self.ques_size, self.config.max_char_length])
                # shape: (batch_size*turn_size, 3, ques_size, 1024)
                _questions_lm = bilm(_questions_char_idxs)

                # shape: (batch_size, para_size, 1024)
                context_elmo = weight_layers("name", context_lm, l2_coef=0.0)
                with tf.variable_scope("", reuse=True):
                    _questions_elmo = weight_layers("name", _questions_lm, l2_coef=0.0)
                    # shape: (batch_size, turn_size, ques_size, 1024)
                    questions_elmo = tf.reshape(_questions_elmo, [self.batch_size, self.turn_size, self.ques_size, 1024])

            # load pretrained cove model from keras
            cove_model = load_model(self.config.cove_word_file)
            # shape: (batch_size, para_size, cove_dim)
            context_cove = cove_model(context_glove)
            _questions_glove = tf.reshape(questions_glove, [questions_batch, self.ques_size, self.config.glove_dim])
            _questions_cove = cove_model(_questions_glove)
            questions_cove = tf.reshape(_questions_cove, [self.batch_size, self.turn_size, self.ques_size, self.config.cove_dim])

            # shape: (batch_size, para_size, [glove_dim(300) + cove_dim(600) + elmo_dim(1024) = embedding_dim])
            c = tf.concat([context_glove, context_cove, context_elmo], axis=-1)
            # shape: (batch_size, turn_size, ques_size, embedding_dim)
            q = tf.concat([questions_glove, questions_cove, questions_elmo], axis=-1)

        with tf.variable_scope("encoding"):
            # context encoding - attention on question
            conf = (self.batch_size, self.turn_size, self.para_size, self.ques_size, self.glove_dim)
            # shape: (batch_size, turn_size, para_size, glove_dim)
            g = question_attention(context_glove, questions_glove, conf)
            # shape: (batch_size, turn_size, para_size, 1)
            em = tf.expand_dims(self.em, axis=-1)
            # shape: (batch_size, turn_size, para_size, embedding_dim)
            c = tf.tile(tf.expand_dims(c, axis=1), [1, turn_size, 1, 1])

            # shape: (batch_size, turn_size, para_size, [embedding_dim + 1 + glove_dim = total_dim])
            c_0 = tf.concat([c, em, g], axis=-1)

            # question encoding - question integration
            embedding_dim = self.config.glove_dim + self.config.cove_dim + self.config.elmo_dim
            # shape: (batch_size*turn_size, ques_size, embedding_dim)
            _q = tf.reshape(q, [questions_batch, self.ques_size, embedding_dim])

            bi_lstm = rnn(num_layers=2, bidirectional=True, num_units=self.hidden_dim,
                        batch_size=questions_batch, input_size=embedding_dim, is_train=self.is_train)
            # shape: (batch_size*turn_size, ques_size, 4*hidden_dim)
            _q_12 = bi_lstm(_q, seq_len=self.ques_size, concat_layers=True)
            # shape: (batch_size, turn_size, ques_size, 4*hidden_dim)
            q_12 = tf.reshape(_q_12, [batch_size, turn_size, ques_size, 4*self.hidden_dim])
            # shape: (batch_size, turn_size, ques_size, 2*hidden_dim)
            q_1 = tf.slice(q_12, [0, 0, 0, 0], [-1, -1, -1, 2*self.hidden_dim])
            q_2 = tf.slice(q_12, [0, 0, 0, 2*self.hidden_dim], [-1, -1, -1, -1])

            # shape: (batch_size, turn_size, ques_size)
            q_2_fc = tf.squeeze(tf.layers.dense(q_2, 1, use_bias=False), axis=-1)
            q_2_fc = softmax_mask(q_2_fc, self.questions_mask)
            q_2_weights = tf.nn.softmax(q_2_fc)

            # shape: (batch_size, turn_size, 2*hidden_dim)
            q_tilde = tf.squeeze(tf.matmul(tf.expand_dims(q_2_weights, axis=2), q_2), axis=2)

            uni_lstm = rnn(num_layers=1, bidirectional=False, num_units=self.hidden_dim,
                        batch_size=self.batch_size, input_size=2*self.hidden_dim, is_train=self.is_train)
            # shape: (batch_size, turn_size, hidden_dim)
            p = uni_lstm(q_tilde, seq_len=self.turn_size, concat_layers=False)

        with tf.variable_scope("reasoning"):
            # integration-flow x2
            conf = (self.batch_size, self.turn_size, self.para_size, self.total_dim, self.hidden_dim)
            # shape: (batch_size, turn_size, para_size, 3*hidden_dim)
            c_1 = integration_flow(c_0, conf, self.is_train)
            conf = (self.batch_size, self.turn_size, self.para_size, 3*self.hidden_dim, self.hidden_dim)
            c_2 = integration_flow(c_1, conf, self.is_train)

            # attention on question
            # shape: (batch_size, turn_size, para_size, [total_dim + 3*hidden_dim + 3*hidden_dim = c_concat_dim])
            c_concat = tf.concat([c_0, c_1, c_2], axis=-1)
            # shape: (batch_size, turn_size, ques_size, [embedding_dim + 2*hidden_dim + 2*hidden_dim = q_concat_dim)
            q_concat = tf.concat([question____emb, q_1, q_2], axis=-1)

            # shape: (batch_size, turn_size, para_size, hidden_dim)
            conf = (self.batch_size, self.turn_size, self.ques_size, self.config.attention_dim)
            # shape: (batch_size, turn_size, para_size, 2*hidden_dim)
            q_hat = fully_aware_attention(c_concat, q_concat, q_2, conf)

            # integration-flow
            # shape: (batch_size, turn_size, para_size, [3*hidden_dim + 2*hidden_dim])
            c_q_concat = tf.concat([c_2, q_hat], axis=-1)
            conf = (self.batch_size, self.turn_size, self.para_size, 5*self.hidden_dim, self.hidden_dim)
            # shape: (batch_size, turn_size, para_size, 3*hidden_dim)
            c_3 = integration_flow(c_q_concat, conf, self.is_train)

            # attention on context
            # shape: (batch_size, turn_size, para_size, 9*hidden_dim)
            c_concat = tf.concat([c_1, c_2, c_3], axis=-1)
            conf = (self.batch_size, self.turn_size, self.ques_size, self.config.attention_dim)
            # shape: (batch_size, turn_size, para_size, 3*hidden_dim)
            c_hat = fully_aware_attention(c_concat, c_concat, c_3, conf)

            # integration
            # shape: (batch_size, turn_size, para_size, 6*hidden_dim = encoding_dim)
            cc_concat = tf.concat([c_3, c_hat], axis=-1)
            encoding_dim = 6 * self.hidden_dim

            bi_lstm = rnn(num_layers=1, bidirectional=True, num_units=self.hidden_dim,
                        batch_size=questions_batch, input_size=encoding_dim, is_train=self.is_train)
            _cc_concat = tf.reshape(cc_concat, [questions_batch, self.para_size, encoding_dim])
            # shape: (batch_size*turn_size, para_size, encoding_dim)
            _c_4 = bi_lstm(_cc_concat, seq_len=self.para_size, concat_layers=False)
            c_4 = tf.reshape(_c_4, [batch_size, turn_size, para_size, encoding_dim])

        with tf.variable_scope("prediction"):
            # shape: (batch_size, turn_size, encoding_dim)
            p_fc = tf.layers.dense(p, encoding_dim, use_bias=False)

            # shape: (batch_size, turn_size, para_size)
            start_logits = tf.squeeze(tf.matmul(c_4, tf.expand_dims(p_fc, axis=-1)), axis=-1)
            start_probs = tf.nn.softmax(start_logits)

            # shape: (batch_size, turn_size, encoding_dim)
            c_4_avg = tf.squeeze(tf.matmul(tf.expand_dims(start_probs, axis=2), c_4), axis=2)
            gru = tf.contrib.rnn.GruCell(hidden_dim)
            # shape: (batch_size, turn_size, hidden_dim)
            p_hat = gru(p, c_4_avg)
            # shape: (batch_size, turn_size, encoding_dim)
            p_hat_fc = tf.layers.dense(p_hat, encoding_dim, use_bias=False)
            # shape: (batch_size, turn_size, para_size)
            end_logits = tf.squeeze(tf.matmul(c_4, tf.expand_dims(p_hat_fc, axis=-1)), axis=-1)
            end_probs = tf.nn.softmax(end_logits)

            # shape: (batch_size, turn_size, encoding_dim)
            c_4_sum = tf.reduce_sum(c_4, axis=2)
            c_4_max = tf.reduce_max(c_4, axis=2)
            # shape: (batch_size, turn_size, 2*encoding_dim)
            c_4_concat = tf.concat([c_4_sum, c_4_max], axis=-1)
            # shape: (batch_size, turn_size, 2*encoding_dim)
            unk_answer_p_fc = tf.layers.dense(p, 2*encoding_dim, use_bias=False)
            # shape: (batch_size, turn_size)
            unk_answer_logits = tf.squeeze(tf.matmul(tf.expand_dims(c_4_concat, axis=2),
                                             tf.expand_dims(unk_answer_p_fc, axis=-1)), axis=-1)
            unk_answer_probs = tf.nn.sigmoid(unk_answer_logits)

            # shape: (batch_size, turn_size, 2*encoding_dim)
            yes_answer_p_fc = tf.layers.dense(p, 2*encoding_dim, use_bias=False)
            # shape: (batch_size, turn_size)
            yes_answer_logits = tf.squeeze(tf.matmul(tf.expand_dims(c_4_concat, axis=2),
                                             tf.expand_dims(yes_answer_p_fc, axis=-1)), axis=-1)
            yes_answer_probs = tf.nn.sigmoid(yes_answer_logits)

            # shape: (batch_size, turn_size, 2*encoding_dim)
            no_p_fc = tf.layers.dense(p, 2*encoding_dim, use_bias=False)
            # shape: (batch_size, turn_size)
            no_answer_logits = tf.squeeze(tf.matmul(tf.expand_dims(c_4_concat, axis=2),
                                             tf.expand_dims(no_answer_p_fc, axis=-1)), axis=-1)
            no_answer_probs = tf.nn.sigmoid(no_answer_logits)

        with tf.name_scope("loss"):
            _start_logits = tf.reshape(start_logits, [-1, self.para_size])
            _end_logits = tf.reshape(end_logits, [-1, self.para_size])
            context_mask = tf.tile(tf.expand_dims(self.context_mask, axis=1), [1, self.turn_size, 1])
            _start_logits_masked = softmax_mask(_start_logits, context_mask)
            _end_logits_masked = softmax_mask(_end_logits, context_mask)
            _starts = tf.reshape(starts, [-1, self.para_size])
            _ends = tf.reshape(ends, [-1, self.para_size])

            span_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=_start_logits_masked, labels=_starts) + \
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=_end_logits_masked, labels=_ends)
            _turn_mask = tf.reshape(self.turn_mask, [-1])
            span_losses_masked = tf.boolean_mask(span_losses, _turn_mask)
            span_loss = tf.reduce_mean(span_losses_masked)

            # yes/no questions
            _unk_answer_logits = tf.reshape(_unk_answer_logits, [-1])
            _unk_answer_logits_masked = tf.boolean_mask(_unk_answer_logits, _turn_mask)
            _unk_answers = tf.reshape(self.unk_answers, [-1])
            _unk_answers_masked = tf.boolean_mask(_unk_answers, _turn_mask)

            _yes_answer_logits = tf.reshape(yes_answer_logits, [-1])
            _yes_answer_logits_masked = tf.boolean_mask(_yes_answer_logits, _turn_mask)
            _yes_answers = tf.reshape(self.yes_answers, [-1])
            _yes_answers_masked = tf.boolean_mask(_yes_answers, _turn_mask)

            _no_answer_logits = tf.reshape(no_answer_logits, [-1])
            _no_answer_logits_masked = tf.boolean_mask(_no_answer_logits, _turn_mask)
            _no_answers = tf.reshape(self.no_answers, [-1])
            _no_answers_masked = tf.boolean_mask(_no_answers, _turn_mask)

            yes_no_unk_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=_yes_answer_logits_masked, labels=_yes_answers_masked) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=_no_answer_logits_masked, labels=_no_answers_masked) + \
                tf.nn.sigmoid_cross_entropy_with_logits(logits=_unk_answer_logits_masked, labels=_unk_answers_masked)
            yes_no_unk_loss = tf.reduce_mean(yes_no_unk_losses)
            span_loss = tf.cond(tf.equal(span_flag, 1), span_loss, 0)
            self.loss = yes_no_unk_loss + span_loss
