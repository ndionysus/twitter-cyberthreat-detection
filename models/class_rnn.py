import tensorflow as tf
import numpy as np
from models.tf_model import TFModel


class RNN(TFModel):
    def input_layer(self):
        '''
            Data and Hyperparameters
        '''
        with tf.variable_scope("input_layer"):
            # Tensor containing word ids
            # shape = (batch size, max length of sentence in batch)
            self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_ids")
            # Tensor containing the real length of each sentence
            # shape = (batch size)
            self.sentence_lengths = tf.placeholder(tf.int32, shape=[None],
                                                   name="sentence_lengths")
            # Tensor containing char ids
            # shape = (batch size, max length of sentence, max length of word)
            self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                           name="char_ids")

            # shape = (batch_size, max_length of sentence)
            self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                               name="word_lengths")

            # shape = (batch size, one-hot)
            self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                         name="labels")

            # Dropout tensors
            self.char_drop_input = tf.placeholder_with_default(
                input=1.0, shape=(), name="char_drop_input")
            self.char_drop_state = tf.placeholder_with_default(
                input=1.0, shape=(), name="char_drop_state")
            self.char_drop_output = tf.placeholder_with_default(
                input=1.0, shape=(), name="char_drop_output")
            self.word_drop_input = tf.placeholder_with_default(
                input=1.0, shape=(), name="word_drop_input")
            self.word_drop_state = tf.placeholder_with_default(
                input=1.0, shape=(), name="word_drop_state")
            self.word_drop_output = tf.placeholder_with_default(
                input=1.0, shape=(), name="word_drop_output")

            # Training variables
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # Using a decaying learning rate
            self.lr = tf.train.exponential_decay(
                learning_rate=self.config.learning["rate"],
                global_step=self.global_step,
                decay_steps=self.config.learning["decay_steps"],
                decay_rate=self.config.learning["decay"],
                staircase=self.config.learning["staircase"])
            # Create the optimizer/trainer
            # I initialize it here for multi-gpu training
            self.optimizer = tf.train.AdamOptimizer(self.lr)

    def embedding_layer(self):
        '''
            Embedding matrices
        '''
        with tf.variable_scope("embedding_layer"):
            if self.config.pretrained is None:
                # Using randomly initialized vectors
                # Word embedding matrix
                word_embedding = tf.get_variable(
                    name="word_embedding",
                    dtype=tf.float32,
                    initializer=tf.random_uniform(
                        shape=[self.config.n_words, self.config.dim_word],
                        minval=-0.25, maxval=0.25))
            else:
                word_embedding = tf.get_variable(
                    name="word_embedding",
                    initializer=np.asarray(self.config.wordvec_matrix, dtype=np.float32),
                    dtype=tf.float32,
                    trainable=self.config.non_static)

            if self.config.use_chars:
                # Char embedding matrix
                char_embedding = tf.get_variable(
                    name="char_embedding",
                    dtype=tf.float32,
                    initializer=tf.random_uniform(
                        shape=[self.config.n_chars, self.config.dim_char],
                        minval=-0.25, maxval=0.25))

            self.word_vectors = tf.nn.embedding_lookup(
                word_embedding, self.word_ids, name="word_matrix")

            if self.config.use_chars:
                self.char_vectors = tf.nn.embedding_lookup(
                    char_embedding, self.char_ids, name="char_matrix")

            '''
                word_embedding = (batch size, max length of sentence in batch, self.config.dim_word)
                char_embedding = (batch size, max length of sentence in batch, max length of word, self.config.dim_char)
            '''

    def RNN_layer(self):
        '''
            Recurrent Layer
        '''

        def Cells(num_units, char_cell=False):
            '''
                Function to build cells
            '''
            # TODO: Redo

            if self.config.model == "rnn":
                self.cell_fw = tf.contrib.rnn.BasicRNNCell(num_units=num_units)
                if self.config.bidirectional:
                    self.cell_bw = tf.contrib.rnn.BasicRNNCell(num_units=num_units)

            elif self.config.model == "lstm":
                self.cell_fw = tf.contrib.rnn.LSTMCell(num_units=num_units)
                if self.config.bidirectional:
                    self.cell_bw = tf.contrib.rnn.LSTMCell(num_units=num_units)

            else:
                self.cell_fw = tf.contrib.rnn.GRUCell(num_units=num_units)
                if self.config.bidirectional:
                    self.cell_bw = tf.contrib.rnn.GRUCell(num_units=num_units)

            if char_cell:
                self.cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell=self.cell_fw, input_keep_prob=self.char_drop_input, output_keep_prob=self.char_drop_output, state_keep_prob=self.char_drop_state)
                if self.config.bidirectional:
                    self.cell_bw = tf.contrib.rnn.DropoutWrapper(
                        cell=self.cell_bw, input_keep_prob=self.char_drop_input, output_keep_prob=self.char_drop_output, state_keep_prob=self.char_drop_state)
            else:
                self.cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell=self.cell_fw, input_keep_prob=self.word_drop_input, output_keep_prob=self.word_drop_output, state_keep_prob=self.word_drop_state)
                if self.config.bidirectional:
                    self.cell_bw = tf.contrib.rnn.DropoutWrapper(
                        cell=self.cell_bw, input_keep_prob=self.word_drop_input, output_keep_prob=self.word_drop_output, state_keep_prob=self.word_drop_state)

        # Word Level Network
        if self.config.use_chars:
            with tf.variable_scope("word_layer"):
                # Put the word length in the axis 1 (time dimension)
                s = tf.shape(self.char_vectors)
                # new shape = [batch*sentence_length,word_length,char_dim]
                self.char_vectors = tf.reshape(self.char_vectors,
                                               shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # CELLS
                Cells(self.config.cell_char)

                # Bidirectional
                if self.config.bidirectional:
                    _, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=self.cell_fw, cell_bw=self.cell_bw, inputs=self.char_vectors,
                        sequence_length=word_lengths, dtype=tf.float32)

                    if self.config.model == "lstm":
                        output_state_fw, output_state_bw = output_state_fw[1], output_state_bw[1]

                    self.char_output = tf.concat([output_state_fw, output_state_bw], axis=-1)

                # Unidirectional
                else:
                    _, output_state_fw = tf.nn.dynamic_rnn(
                        cell=self.cell_fw, inputs=self.char_vectors,
                        sequence_length=word_lengths, dtype=tf.float32)

                    if self.config.model == "lstm":
                        output_state_fw = output_state_fw[1]

                    self.char_output = output_state_fw

                # shape = (batch size, max sentence length, char hidden size)
                self.h = self.char_output.shape[1].value
                self.char_output = tf.reshape(self.char_output, shape=[s[0], s[1], self.h])
                self.word_vectors = tf.concat([self.word_vectors, self.char_output], axis=-1)

        # Sentence Level Network
        with tf.variable_scope("sentence_layer"):

            # Create Cells
            Cells(self.config.cell_word)

            # Bidirectional
            if self.config.bidirectional:
                _, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.cell_fw, cell_bw=self.cell_bw, inputs=self.word_vectors,
                    sequence_length=self.sentence_lengths, dtype=tf.float32)

                if self.config.model == "lstm":
                    output_state_fw, output_state_bw = output_state_fw[1], output_state_bw[1]

                self.lstm_output = tf.concat([output_state_fw, output_state_bw], axis=-1)

            # Unidirectional
            else:
                _, output_state_fw = tf.nn.dynamic_rnn(
                    cell=self.cell_fw, inputs=self.word_vectors,
                    sequence_length=self.sentence_lengths, dtype=tf.float32)

                if self.config.model == "lstm":
                    output_state_fw = output_state_fw[1]

                self.lstm_output = output_state_fw

            self.h = self.lstm_output.shape[1].value

    def output_layer(self):
        with tf.variable_scope("output_layer"):
            layer = {'weights': tf.get_variable(name="output_w", shape=[self.h, 2],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                     'biases': tf.get_variable(name="output_b", initializer=tf.constant(0.1, shape=[2]))}
            self.scores = tf.nn.xw_plus_b(
                self.lstm_output, layer["weights"], layer["biases"], name="scores")

    def loss_function(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                               logits=self.scores)
        self.loss = tf.reduce_mean(self.loss)

    def acc_function(self):
        with tf.variable_scope("prediction_layer"):
            self.predictions = tf.argmax(self.scores, axis=1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

    def train_op(self):
        with tf.variable_scope("train_step"):
            self.gradient = self.optimizer.compute_gradients(loss=self.loss)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars=self.gradient,
                                                           global_step=self.global_step)

    def build(self):
        self.input_layer()
        self.embedding_layer()
        self.RNN_layer()
        self.output_layer()
        self.loss_function()
        self.acc_function()
        # Generic functions that add training op and initialize session
        self.train_op()
        self.initialize_session()  # now self.sess is defined and vars are init

    def load_model(self, dir):
        self.initialize_session()
        self.saver = tf.train.import_meta_graph("{}.meta".format(dir))
        self.saver.restore(self.sess, dir)

        # Get the operations easily
        graph = tf.get_default_graph()
        # INPUT_LAYER
        self.word_ids = graph.get_operation_by_name("input_layer/word_ids").outputs[0]
        self.sentence_lengths = graph.get_operation_by_name(
            "input_layer/sentence_lengths").outputs[0]
        self.char_ids = graph.get_operation_by_name("input_layer/char_ids").outputs[0]
        self.word_lengths = graph.get_operation_by_name("input_layer/word_lengths").outputs[0]
        self.labels = graph.get_operation_by_name("input_layer/labels").outputs[0]
        # OUTPUT_LAYER
        self.scores = graph.get_operation_by_name("output_layer/scores").outputs[0]
        self.predictions = graph.get_operation_by_name("prediction_layer/predictions").outputs[0]
        self.accuracy = graph.get_operation_by_name("prediction_layer/accuracy").outputs[0]

        print("\nModel as been loaded..")

    def __init__(self, config):
        super(RNN, self).__init__(config)
