import tensorflow as tf
import numpy as np
from models.tf_model import TFModel


class CNN(TFModel):
    def input_layer(self):
        '''
            Data and Hyperparameters
        '''
        with tf.variable_scope("input_layer"):
            # Tensor containing word ids
            # shape = (batch size, max length of sentence in batch)
            self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_ids")
            # shape = (batch size)
            self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                         name="labels")

            # Other Hyper parameters
            self.dropout = tf.placeholder_with_default(input=1.0, shape=(), name="dropout")

            # Training variables
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # Track of l2 regularization loss
            self.regularizers = tf.constant(0.0)
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
                    initializer=tf.random_uniform([self.config.n_words, self.config.dim_word],
                                                  minval=-0.25,
                                                  maxval=0.25))
            else:
                word_embedding = tf.get_variable(
                    name="word_embedding",
                    initializer=np.asarray(self.config.wordvec_matrix, dtype=np.float32),
                    dtype=tf.float32,
                    trainable=self.config.non_static)

            self.word_vectors = tf.nn.embedding_lookup(
                word_embedding, self.word_ids, name="word_matrix")
            self.word_vectors = tf.expand_dims(self.word_vectors, 1)

    def CNN(self):
        with tf.name_scope("convolutional_neural_network"):
            pooled_outputs = []

            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.variable_scope("ConvMax-%s" % i):
                     # Convolutional Layer
                    # Set Filter shape
                    self.filter_shape = [self.config.n_channels, filter_size,
                                         self.config.dim_word, self.config.n_filters]
                    # Set weights and biases
                    W = tf.get_variable(
                        name="ConvMax_w", initializer=tf.truncated_normal(self.filter_shape))
                    b = tf.get_variable(name="ConvMax_b",
                                        initializer=tf.constant(0.0, shape=[self.config.n_filters]))
                    # Create Convolutional model
                    self.conv = tf.nn.convolution(self.word_vectors, W, padding="VALID", dilation_rate=[
                                                  1, 1], name="Convolution")
                    # Add the biases
                    self.conv = tf.nn.bias_add(self.conv, b)
                    # Apply Activation function/ nonlinearity
                    self.conv = tf.nn.relu(self.conv, name="ReLU")

                    # Max Pooling Layer
                    self.max_pool = tf.reduce_max(self.conv, axis=2, keepdims=True)
                    pooled_outputs.append(self.max_pool)

            # Combine all the pooled features
            # Get the final number of features
            self.total_features = self.config.n_filters * len(self.config.filter_sizes)
            # Combine all layers
            self.features = tf.concat(values=pooled_outputs, axis=3)
            # Reshape it to form a one dimensional tensor
            self.features = tf.reshape(self.features, [-1, self.total_features])
            self.model = tf.nn.dropout(self.features, self.dropout)

    def CNN_original(self):
        with tf.name_scope("convolutional_neural_network"):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.variable_scope("ConvMax-%s" % filter_size):
                     # Convolutional Layer
                    # Set Filter shape
                    self.filter_shape = [filter_size, self.config.dim_word,
                                         self.config.n_channels, self.config.n_filters]
                    # Set weights and biases
                    W = tf.get_variable(
                        name="ConvMax_w%d" % i, initializer=tf.truncated_normal(self.filter_shape))
                    b = tf.get_variable(name="ConvMax_b%d" % filter_size,
                                        initializer=tf.constant(0.0, shape=[self.config.n_filters]))
                    # Create Convolutional model
                    self.conv = tf.nn.convolution(self.word_vectors, W, padding="VALID", dilation_rate=[
                                                  1, 1], name="Convolution")

                    # Add the biases
                    model = tf.nn.bias_add(self.conv, b)
                    # Apply Activation function/ nonlinearity
                    model = tf.nn.relu(model, name="ReLU")
                    # Max Pooling Layer
                    model = tf.reduce_max(model, axis=1, keepdims=True)
                    pooled_outputs.append(model)

            # Combine all the pooled features
            # Get the final number of features
            self.total_features = self.config.n_filters * len(self.config.filter_sizes)
            # Combine all layers
            features = tf.concat(values=pooled_outputs, axis=3)
            # Reshape it to form a one dimensional tensor
            features = tf.reshape(features, [-1, self.total_features])
            self.model = tf.nn.dropout(features, self.dropout)

    def FCNN(self):
        with tf.name_scope("fully_connected_neural_network"):
            # Build the Fully Connected neural network
            for l, nodes in enumerate(self.config.fcnn_layer):
                layer = {
                    'weights': tf.get_variable(name="fcnn_w%d" % l, initializer=tf.truncated_normal([self.total_features, nodes])),
                    'biases': tf.get_variable(name="fcnn_b%d" % l, initializer=tf.truncated_normal([nodes]))
                }
                self.total_features = nodes
                self.model = tf.nn.xw_plus_b(self.model, layer["weights"], layer["biases"])
                self.model = tf.nn.relu(self.model)
                self.regularizers += tf.nn.l2_loss(layer["weights"])
                self.regularizers += tf.nn.l2_loss(layer["biases"])

    def output_layer(self):
        with tf.variable_scope("output_layer"):
            layer = {'weights': tf.get_variable(name="output_w", shape=[self.total_features, 2],
                                                initializer=tf.contrib.layers.xavier_initializer()),
                     'biases': tf.get_variable(name="output_b", initializer=tf.constant(0.1, shape=[2]))}
            self.scores = tf.nn.xw_plus_b(
                self.model, layer["weights"], layer["biases"], name="scores")
            self.regularizers += tf.nn.l2_loss(layer["weights"])
            self.regularizers += tf.nn.l2_loss(layer["biases"])

    def loss_function(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                               logits=self.scores)
        self.loss = tf.reduce_mean(self.loss) + self.config.l2_reg_lambda * self.regularizers

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
        self.CNN()
        self.FCNN()
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
        self.labels = graph.get_operation_by_name("input_layer/labels").outputs[0]
        # OUTPUT_LAYER
        self.scores = graph.get_operation_by_name("output_layer/scores").outputs[0]
        self.predictions = graph.get_operation_by_name("prediction_layer/predictions").outputs[0]
        self.accuracy = graph.get_operation_by_name("prediction_layer/accuracy").outputs[0]

        print("\nModel as been loaded..")

    def __init__(self, config):
        super(CNN, self).__init__(config)

    '''
        Non-Tensorflow Operations
    '''
