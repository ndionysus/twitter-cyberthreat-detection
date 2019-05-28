import utils
import numpy as np
import datetime


class Config():
    def load_datasets(self):
        # Train set
        self.train = self.build_dataset(pos=self.train["pos"], neg=self.train["neg"])
        self.train = utils.shuffle(self.train)

        # Test sets
        if not self.test == None:
            test = self.test
            self.test = []
            for pos, neg in zip(test["pos"], test["neg"]):
                self.test.append(self.build_dataset(pos=pos, neg=neg))

    def pad(self, dataset, labels=True):
        def add_padding(words, pad, chars=None, charpad=None):
            # Need to store the real word length if we are using chars
            word_length = []
            # For each sentence
            for i in range(len(words)):
                words[i] = (list(words[i]) + [0] * pad)[:pad]
                if not len(chars) == 0:
                    wlen = []
                    chars[i] = (list(chars[i]) + [[0]] * pad)[:pad]
                    for j in range(len(chars[i])):
                        wlen.append(len(chars[i][j]))
                        chars[i][j] = (chars[i][j] + ([0] * charpad))[:charpad]
                    word_length.append(wlen)
            return words, zip(chars, word_length)

        # We will return the data separated, ready to be fed
        tensor_words = []
        sentence_length = []
        tensor_chars = []

        maxlen_sentence = 0
        maxlen_word = 0

        if labels:
            text, labels = zip(*dataset)
        else:
            text = dataset

        for line in text:
            sentence_length.append(len(line))
            if len(line) > maxlen_sentence:
                maxlen_sentence = len(line)
            if self.use_chars:
                words, chars = zip(*line)
                local_maxlen_word = (max(len(c) for c in chars))
                # Maximum sentence length is already doen above, now we need
                # max word length
                if local_maxlen_word > maxlen_word:
                    maxlen_word = local_maxlen_word
                tensor_chars.append(chars)
            else:
                # Convinience...
                words = line
            tensor_words.append(words)

        tensor_words, tensor_chars = add_padding(words=tensor_words, pad=maxlen_sentence,
                                                 chars=tensor_chars, charpad=maxlen_word)

        return zip(tensor_words, sentence_length), tensor_chars, labels

    def feed(self, model, words, labels, chars=None, dropout=1.0):
        words, sentence_length = zip(*words)

        feed = {
            model.word_ids: words
        }

        if not labels == False:
            feed[model.labels] = labels

        # Convolutional Neural Network
        if self.model == "cnn":
            if not dropout == 1.0:
                feed[model.dropout] = dropout

        # Recurrent Neural Network
        else:
            feed[model.sentence_lengths] = sentence_length

            if not dropout == 1.0:
                feed[model.char_drop_input] = dropout["char_input"]
                feed[model.char_drop_state] = dropout["char_state"]
                feed[model.char_drop_output] = dropout["char_output"]
                feed[model.word_drop_input] = dropout["word_input"]
                feed[model.word_drop_state] = dropout["word_state"]
                feed[model.word_drop_output] = dropout["word_output"]

            if self.use_chars:
                chars, word_lengths = zip(*chars)
                feed[model.char_ids] = chars
                feed[model.word_lengths] = word_lengths

        return feed

    def word_id(self, word):
        if word in self.words:
            w_id = self.words[word]
        else:
            w_id = self.words["<UNK>"]

        c_ids = []
        if self.use_chars:
            for c in word:
                if c in self.chars:
                    c_ids.append(self.chars[c])
                else:
                    c_ids.append(self.chars["<UNK>"])
            w_id = [w_id, c_ids]
        return w_id

    def build_dataset(self, pos, neg):
        # Check if files exist
        if not (utils.check_file(pos) and utils.check_file(neg)):
            raise Exception("Files ", pos, " and ", neg, " do not exist")

        pos = list(open(pos, "r").readlines())
        neg = list(open(neg, "r").readlines())
        # TODO: Deprecate full clean
        if self.full_clean:
            pos = utils.full_clean(dataset=pos)
            neg = utils.full_clean(dataset=neg)
        # Create labels
        pos_labels = [[0, 1] for _ in pos]
        neg_labels = [[1, 0] for _ in neg]
        # Combine sets
        text = pos + neg
        labels = pos_labels + neg_labels
        # Build set
        for i, line in enumerate(text):
            text[i] = []
            line = line.strip().split(" ")
            for w in line:
                text[i].append(self.word_id(w))
            text[i] = [text[i], labels[i]]

        return np.array(text)

    def build_vocabs(self):
        # Build the word vocab
        self.words = dict()

        # Assign the Unknown token
        self.words["<UNK>"] = 0

        # Build the character vocab
        if self.use_chars:
            self.chars = dict()
            # Assign the Unknown token
            self.chars["<UNK>"] = 0
        else:
            # Need to initialize the character vocab either way
            self.chars = None

        # If we have pretrained vectors, build them here
        if not self.pretrained == None:
            # Load the model
            print("USING: Pre-trained embedding vectors from %s" % self.pretrained)
            vectors = np.load(self.pretrained)
            vectors = {key: vectors[key].item() for key in vectors}["embeddings"]
            # To prevent mistakes regarding embedding dimensions, take the
            # "first" value in the dict and check the embedding size
            self.dim_word = len(list(vectors.values())[0])
            # Embedding matrix
            self.wordvec_matrix = []
            # <UNK> vector
            self.wordvec_matrix.append(np.random.uniform(low=-0.25, high=0.25, size=self.dim_word))
        else:
            print("USING: Randomly initialized vectors")

        # Load and join positive and negative files
        pos = list(open(self.train["pos"], "r").readlines())
        neg = list(open(self.train["neg"], "r").readlines())
        dset = pos + neg

        # If using complete clean
        if self.full_clean:
            print("USING: Full clean")
            dset = utils.full_clean(dataset=dset)

        # If using dynamic padding
        if self.dynamic_pad:
            self.maxlen_sentence = 0
            print("USING: Dynamic padding")
        else:
            self.maxlen_sentence = len(max(dset, key=len).split(" "))
            print("USING: Padding is set to a maximum of ", self.maxlen_sentence)

        # Build the embedding matrix
        # word and character counter
        nw, nc = 1, 1
        for line in dset:
            line = line.strip().split(" ")
            for w in line:
                if not w in self.words:
                    self.words[w] = nw
                    # If using pre-trained vectors
                    if not self.pretrained == None:
                        # If word is in the embeddings dictionary
                        if w in vectors:
                            self.wordvec_matrix.append(vectors[w])
                        else:
                            self.wordvec_matrix.append(np.random.uniform(
                                low=-0.25, high=0.25, size=self.dim_word))
                    nw += 1
                    if self.use_chars:
                        for c in w:
                            if not c in self.chars:
                                self.chars[c] = nc
                                nc += 1

    def data(self):
        infra = "A"
        self.train = {
            "pos": "./data/d1_" + infra + "_pos.txt",
            "neg": "./data/d1_" + infra + "_neg.txt"
        }
        self.test = {
            "pos": ["./data/d2_" + infra + "_pos.txt",
                     "./data/d3_" + infra + "_pos.txt"],
            "neg": ["./data/d2_" + infra + "_neg.txt",
                     "./data/d3_" + infra + "_neg.txt"]
        }

    def load_data(self):
        self.build_vocabs()
        self.n_words = len(self.words)
        if self.use_chars:
            self.n_chars = len(self.chars)
        self.load_datasets()

    def embedding(self, args):
        self.pretrained = args.embeddings
        self.non_static = True
        self.use_chars = args.chars
        self.dim_char = 300
        self.dim_word = 300

    def training(self, args):
        self.save = args.save
        self.full_clean = args.clean
        self.dynamic_pad = args.padding
        self.k_folds = 1
        self.test_split = 0.0
        self.n_epochs = 200
        self.batch_size = 256
        self.learning = {
            "rate": 0.01,
            "method": "adam",
            "decay": 1,
            "decay_steps": 10,
            "staircase": True
        }

    def cnn(self):
        print("USING: Convolutional Neural Network")
        self.n_channels = 1
        self.n_filters = 300
        self.filter_sizes = [2, 3, 4, 5, 6, 7]
        self.padding = "VALID"
        self.fcnn_layer = []
        self.l2_reg_lambda = 0.0
        self.dropout = 0.5

    def rnn(self):
        print("USING: Recurrent Neural Network")
        self.cells = self.model
        self.bidirectional = True
        self.cell_char = 25
        self.cell_word = 100
        self.dropout = {
            "char_input": 1.0,
            "char_state": 1.0,
            "char_output": 0.5,
            "word_input": 1.0,
            "word_state": 1.0,
            "word_output": 0.5,
        }

    def __init__(self, args=None, load=False):
        if not load:
            self.model = args.model
            self.embedding(args)
            self.training(args)
            self.data()
            self.load_data()
            if self.model == "cnn":
                self.cnn()
            elif self.model in ["rnn", "lstm", "gru"]:
                self.rnn()
            else:
                raise ValueError("No model named %s" % self.model)

        else:
            print("Loading Model from ", args.model)
            config = np.load(args.model + "config.npz",allow_pickle=True)
            # Store the dir
            self.dir = args.model
            # Get the word dictionary
            self.words = {key: config[key].item() for key in config}["words"]
            # Get the character dictionary
            self.chars = {key: config[key].item() for key in config}["chars"]
            if self.chars == None:
                self.use_chars = False
            else:
                self.use_chars = True
            # Get other info
            self.k_folds = config["k_folds"]
            self.model = config["model"]
            self.save = None

            # TODO: Will be deprecated
            self.full_clean = False
