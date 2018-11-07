import tensorflow as tf


class rnn:
    def __init__(self, 
            num_layers,
            bidirectional,
            num_units,
            batch_size,
            input_size,
            keep_prob=1.0,
            is_train=None,
            type="lstm"):

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnns = []
        self.inits = []
        self.dropout_mask = []
        self.scope = type
        if self.scope == "lstm":
            self.rnn_cell = tf.contrib.rnn.LSTMCell
        elif self.scope == "gru":
            self.rnn_cell = tf.contrib.rnn.GRUCell
        else:
            self.rnn_cell = tf.contrib.rnn.RNNCell

        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units if self.bidirectional else num_units
            rnn_fw = self.rnn_cell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                            keep_prob=keep_prob, is_train=is_train, mode=None)
            if self.bidirectional:
                rnn_bw = self.rnn_cell(num_units)
                init_bw = tf.tile(tf.Variable(
                    tf.zeros([1, num_units])), [batch_size, 1])
                mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                keep_prob=keep_prob, is_train=is_train, mode=None)
                self.rnns.append((rnn_fw, rnn_bw, ))
                self.inits.append((init_fw, init_bw, ))
                self.dropout_mask.append((mask_fw, mask_bw, ))
            else:
                self.rnns.append((rnn_fw, ))
                self.inits.append((init_fw, ))
                self.dropout_mask.append((mask_fw, ))
                
    def __call__(self,
            inputs,
            seq_len,
            keep_prob=1.0,
            is_train=None,
            concat_layers=True):

        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                if self.bidirectional:
                    rnn_fw, rnn_bw = self.rnns[layer]
                    init_fw, init_bw = self.inits[layer]
                    mask_fw, mask_bw = self.dropout_mask[layer]
                    with tf.variable_scope("fw_{}".format(layer)):
                        out_fw, _ = tf.nn.dynamic_rnn(
                            rnn_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                    with tf.variable_scope("bw_{}".format(layer)):
                        inputs_bw = tf.reverse_sequence(
                            outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                        out_bw, _ = tf.nn.dynamic_rnn(
                            rnn_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                        out_bw = tf.reverse_sequence(
                            out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    outputs.append(tf.concat([out_fw, out_bw], axis=2))
                else:
                    rnn_fw = self.rnns[layer]
                    init_fw = self.rnns[layer]
                    mask_fw = self.dropout_mask[layer]
                    with tf.variable_scope("fw_{}".format(layer)):
                        out_fw, _ = tf.nn.dynamic_rnn(
                            rnn_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                    outputs.append(out_fw)

        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


def question_attention(context, questions, conf):
    """
    Parameters
    :context, (batch_size, para_size, glove_dim)
    :questions, (batch_size, turn_size, ques_size, glove_dim)
    :shape, configuration

    Returns
    :res, (batch_size, turn_size, para_size, glove_dim)
    """
    batch_size, turn_size, para_size, ques_size, glove_dim = conf

    # shape: (batch_size, para_size, glove_dim)
    context_fc = tf.layers.dense(context, glove_dim, activation=tf.nn.relu, use_bias=False)

    questions_fc = tf.layers.dense(questions, glove_dim, activation=tf.nn.relu, use_bias=False)
    _questions_fc = tf.reshape(questions_fc, [batch_size, turn_size*ques_size, glove_dim])

    # shape: (batch_size, para_size, turn_size*ques_size)
    _outputs = tf.matmul(context_fc, tf.transpose(_questions_fc, [0, 2, 1]))
    # shape: (batch_size, para_size, turn_size, ques_size)
    outputs = tf.reshape(_outputs, [batch_size, para_size, turn_size, ques_size])

    # shape: (batch_size, para_size, turn_size, ques_size)
    weights = tf.nn.softmax(outputs)
    # shape: (batch_size, turn_size, para_size, ques_size)
    weights = tf.transpose(weights, [0, 2, 1, 3])
    # shape: (batch_size, turn_size, para_size, glove_dim)
    res = tf.matmul(weights, questions)
    return res


def fully_aware_attention(context_how, questions_how, questions, conf):
    """
    :context_how, (batch_size, turn_size, para_size, c_concat_dim)
    :questions_how, (batch_size, turn_size, ques_size, q_concat_dim)
    :questions, (batch_size, turn_size, ques_size, ques_dim)

    :res, (batch_size, turn_size, para_size, ques_dim)
    """
    batch_size, turn_size, ques_size, attention_dim = conf

    # shape: (batch_size, turn_size, para_size, attention_dim)
    context_fc = tf.layers.dense(context_how, attention_dim, activation=tf.nn.relu, use_bias=False)

    # shape: (batch_size, turn_size, ques_size, attention_dim)
    questions_fc = tf.layers.dense(questions_how, attention_dim, activation=tf.nn.relu, use_bias=False)
    diagonal = tf.get_variable("diagonal", shape=[attention_dim], dtype=tf.float32)
    # shape: (attention_dim, attention_dim)
    diagonal_mat = tf.linalg.tensor_diag(diagonal)
    # shape: (*, attention_dim)
    flat_questions = tf.reshape(questions_fc, [-1, attention_dim])
    # shape: (*, attention_dim)
    flat_questions_fc = tf.matmul(flat_questions, diagonal_mat)
    questions = tf.reshape(flat_questions_fc, [batch_size, turn_size, ques_size, attention_dim])
    # shape: (batch_size, turn_size, para_size, ques_size)
    outputs = tf.matmul(context_fc, tf.transpose(questions, [0, 1, 3, 2]))
    weights = tf.nn.softmax(outputs)

    # shape: (batch_size, turn_size, para_size, embedding_dim)
    res = tf.matmul(weights, questions)
    return res


def integration_flow(context, conf, is_train):
    """
    :context, (batch_size, turn_size, para_size, total_dim)

    Returns
    :c_next, (batch_size, turn_size, para_size, 3*hidden_dim)
    """
    #TODO:
    batch_size, turn_size, para_size, total_dim, hidden_dim = conf

    _context = tf.reshape(context, [batch_size*turn_size, para_size, total_dim])

    integration = rnn(num_layers=1, bidirectional=True, num_units=hidden_dim,
        batch_size=batch_size*turn_size, input_size=total_dim, is_train=is_train)

    # shape: (batch_size*turn_size, para_size, 2*hidden_dim)
    c_hat = integration(_context, seq_len=para_size)
    c_hat = tf.reshape(c_hat, [batch_size, turn_size, para_size, 2*hidden_dim])
    _c_hat = tf.transpose(c_hat, [0, 2, 1, 3])
    _c_hat = tf.reshape(_c_hat, [batch_size*para_size, turn_size, 2*hidden_dim])

    flow = rnn(num_layers=1, bidirectional=False, num_units=hidden_dim,
        batch_size=batch_size*para_size, input_size=2*hidden_dim, is_train=is_train)

    # shape: (batch_size*para_size, turn_size, hidden_dim)
    f = flow(_c_hat)
    f = tf.reshape(f, [batch_size, para_size, turn_size, hidden_dim])
    f = tf.transpose(f, [0, 2, 1, 3])

    # shape: (batch_size, para_size, turn_size, 3*hidden_dim)
    c_next = tf.concat([c_hat, f], axis=-1)
    return c_next


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args
