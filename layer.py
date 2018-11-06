import tensorflow as tf


class cudnn_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


class bi_lstm:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="lstm"):
        self.num_layers = num_layers
        self.lstms = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            lstm_fw = tf.contrib.rnn.LSTMCell(num_units)
            lstm_bw = tf.contrib.rnn.LSTMCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.lstms.append((lstm_fw, lstm_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                lstm_fw, lstm_bw = self.lstms[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        lstm_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        lstm_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


class lstm:

    def __init__(self, num_layers, bidirectional, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="lstm"):
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstms = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units if self.bidirectional else num_units
            lstm_fw = tf.contrib.rnn.LSTMCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                            keep_prob=keep_prob, is_train=is_train, mode=None)
            if self.bidirectional:
                lstm_bw = tf.contrib.rnn.LSTMCell(num_units)
                init_bw = tf.tile(tf.Variable(
                    tf.zeros([1, num_units])), [batch_size, 1])
                mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                keep_prob=keep_prob, is_train=is_train, mode=None)
                self.lstms.append((lstm_fw, lstm_bw, ))
                self.inits.append((init_fw, init_bw, ))
                self.dropout_mask.append((mask_fw, mask_bw, ))
            else:
                self.lstms.append((lstm_fw, ))
                self.inits.append((init_fw, ))
                self.dropout_mask.append((mask_fw, ))
                

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                if self.bidirectional:
                    lstm_fw, lstm_bw = self.lstms[layer]
                    init_fw, init_bw = self.inits[layer]
                    mask_fw, mask_bw = self.dropout_mask[layer]
                    with tf.variable_scope("fw_{}".format(layer)):
                        out_fw, _ = tf.nn.dynamic_rnn(
                            lstm_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                    with tf.variable_scope("bw_{}".format(layer)):
                        inputs_bw = tf.reverse_sequence(
                            outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                        out_bw, _ = tf.nn.dynamic_rnn(
                            lstm_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                        out_bw = tf.reverse_sequence(
                            out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    outputs.append(tf.concat([out_fw, out_bw], axis=2))
                else:
                    lstm_fw = self.lstms[layer]
                    init_fw = self.lstms[layer]
                    mask_fw = self.dropout_mask[layer]
                    with tf.variable_scope("fw_{}".format(layer)):
                        out_fw, _ = tf.nn.dynamic_rnn(
                            lstm_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                    outputs.append(out_fw)

        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


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


def question_attention(context, questions):
    """
    :context, (batch_size, para_limit, embedding_dim)
    :questions, (batch_size, turn_limit, ques_limit, embedding_dim)
    """
    # TODO:
    batch_size = None
    para_limit = None
    ques_limit = None
    turn_limit = None
    embedding_dim = None

    # shape: (batch_size, para_limit, embedding_dim)
    _context = tf.layers.dense(context, embedding_dim, activation=tf.nn.relu, use_bias=False)

    # shape: (B, T*Q, E)
    _questions = tf.layers.dense(questions, embedding_dim, activation=tf.nn.relu, use_bias=False)
    _questions = tf.reshape(questions, [batch_size, turn_limit*ques_limit, embedding_dim])

    # shape: (B, P, T*Q)
    outputs = tf.matmul(_context, tf.transpose(_questions, [0, 2, 1]))
    # shape: (B, P, T, Q)
    outputs = tf.reshape(outputs, [batch_size, para_limit, turn_limit, ques_limit])

    # shape: (B, P, T, Q)
    probs = tf.nn.softmax(outputs)
    # shape: (B, T, P, Q)
    probs = tf.transpose(probs, [0, 2, 1, 3])
    # shape: (B, T, P, E)
    res = tf.matmul(probs, questions)
    return res


def fully_aware_attention(context_how, questions_how, questions):
    """
    :context_how, (batch_size, turn_limit, para_limit, concat_dim)
    :questions_how, (batch_size, turn_limit, ques_limit, concat_dim)
    :questions, (batch_size, turn_limit, ques_limit, embedding_dim)
    """
    k = None
    batch_size = None
    turn_limit = None
    ques_limit = None
    # shape: (batch_size, turn_limit, para_limit, k)
    _context = tf.layers.dense(context_how, k, activation=tf.nn.relu, use_bias=False)

    # shape: (batch_size, turn_limit, ques_limit, k)
    _questions = tf.layers.dense(questions_how, k, activation=tf.nn.relu, use_bias=False)
    diagonal = tf.get_variable("diagonal", shape=[k], dtype=tf.float32)
    # shape: (k, k)
    diagonal_mat = tf.linalg.tensor_diag(diagonal)
    flat_questions = tf.reshape(_questions, [-1, k])
    flat_questions_fc = tf.matmul(flat_questions, diagonal_mat)
    _questions = tf.reshape(flat_questions_fc, [batch_size, turn_limit, ques_limit, k])
    # shape: (batch_size, turn_limit, para_limit, ques_limit)
    outputs = tf.matmul(_context, tf.transpose(_questions, [0, 1, 3, 2]))
    probs = tf.nn.softmax(outputs)

    # shape: (batch_size, turn_limit, para_limit, embedding_dim)
    res = tf.matmul(probs, questions)
    return res


def integration_flow(context, num_units, batch_size, input_size, is_train):
    """
    :context, (batch_size, turn_limit, para_limit, embedding_dim)
    """
    #TODO:
    turn_limit = None
    para_limit = None
    embedding_dim = None

    _context = tf.reshape(context, [batch_size*turn_limit, para_limit, embedding_dim])

    integration = rnn(num_layers=1, bidirectional=True, num_units=num_units,
        batch_size=batch_size*turn_limit, input_size=input_size, is_train=is_train)

    # shape: (batch_size*turn_limit, para_limit, 2*num_units)
    c_hat = integration(_context, seq_len=para_limit)
    c_hat = tf.reshape(c_hat, [batch_size, turn_limit, para_limit, 2*num_units])
    _c_hat = tf.transpose(c_hat, [0, 2, 1, 3])
    _c_hat = tf.reshape(_c_hat, [batch_size*para_limit, turn_limit, 2*num_units])

    flow = rnn(num_layers=1, bidirectional=False, num_units=num_units,
        batch_size=batch_size*para_limit, input_size=input_size, is_train=is_train)

    # shape: (batch_size*para_limit, turn_limit, num_units)
    f = flow(_c_hat)
    f = tf.reshape(f, [batch_size, para_limit, turn_limit, num_units])
    f = tf.transpose(f, [0, 2, 1, 3])

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
