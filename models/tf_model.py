import tensorflow as tf
import numpy as np
import os


class TFModel(object):
    '''
        This class contains the general functions for a tensorflow model
    '''

    def __init__(self, config):
        # Limit the TensorFlow's logs
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
        # tf.logging.set_verbosity(tf.logging.ERROR)

        self.config = config
        self.sess = None
        self.saver = None

    def initialize_session(self):
        """
            Set configurations:
                *   allow_soft_placement : If True, will allow models trained
                                           on GPU to be deployed unto CPU
                *   log_device_placement : If True, will print the hardware
                                           and operations that have been placed on it
        """
        sess_conf = tf.ConfigProto(allow_soft_placement=True,
                                   log_device_placement=False)
        sess_conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_conf)

        # Save object
        if not self.config.save == None:
            self.saver = tf.train.Saver()

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

    def save_model(self, fold, timestamp, name):
        """
            Save the model and the config file
        """
        model_name = name + "_" + timestamp

        main_dir = "./checkpoints/" + model_name + "/"
        # Check main model dir
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)

        # If using K-Fold Cross Validation, save each model
        if self.config.k_folds > 1:
            dir = main_dir + "Fold_" + str(fold + 1) + "/"
            # Create Fold dir
            if not os.path.exists(dir):
                os.makedirs(dir)

            # Save the model
            self.saver.save(self.sess, dir)
        else:
            self.saver.save(self.sess, main_dir)

        return main_dir

    def ner_save(self, fold, timestamp, name, ep):
        # Save the model
        main_dir = self.save_model(fold, timestamp, name)
        # Save the corresponding config file
        if fold == 0:
            np.savez(main_dir + "config",
                     model=self.config.model,
                     k_folds=self.config.k_folds,
                     words=self.config.words,
                     tags=self.config.tags,
                     chars=self.config.chars,
                     use_crf=self.config.use_crf,
                     epoch=ep+1)

    def class_save(self, fold, timestamp, name, ep):
        # Save the model
        main_dir = self.save_model(fold, timestamp, name)
        # Save the config file
        if fold == 0:
            np.savez(main_dir + "config",
                     model=self.config.model,
                     k_folds=self.config.k_folds,
                     words=self.config.words,
                     chars=self.config.chars,
                     epoch=ep+1)

    def close_session(self):
        self.sess.close()
        tf.reset_default_graph()
