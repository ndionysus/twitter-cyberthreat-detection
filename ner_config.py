import utils
import numpy as np
import datetime


class Config():
    def load_datasets(self):
        # Train set
        self.train = self.build_dataset(text_list=self.train["text"], tags_list=self.train["tags"])
        # Test sets
        if not self.test == None:
            test = self.test
            self.test = []
            for text, tags in zip(test["text"], test["tags"]):
                self.test.append(self.build_dataset(text_list=[text], tags_list=[tags]))

    def pad(self, dataset, use_chars, labels=True):
        def add_padding(words, pad, tags=None, chars=None, charpad=None):
            # Need to store the real word length if we are using chars
            word_length = []
            # For each sentence
            for i in range(len(words)):
                words[i] = (list(words[i]) + [0] * pad)[:pad]
                if not len(tags) == 0:
                    tags[i] = (list(tags[i]) + [0] * pad)[:pad]
                if not len(chars) == 0:
                    wlen = []
                    chars[i] = (list(chars[i]) + [[0]] * pad)[:pad]
                    for j in range(len(chars[i])):
                        wlen.append(len(chars[i][j]))
                        chars[i][j] = (chars[i][j] + ([0] * charpad))[:charpad]
                    word_length.append(wlen)

            return words, tags, zip(chars, word_length)

        # We will return the data separated, ready to be fed
        tensor_words = []
        sentence_length = []
        tensor_tags = []
        tensor_chars = []

        # In NER, since each word has a label we have to iterate each sentence
        maxlen_sentence = 0
        maxlen_word = 0

        for sentence in dataset:
            # Arrays for storing
            # Seperate labels from input
            if labels:
                input, tags = zip(*sentence)
            else:
                input = sentence
            # Use the labels to find the maximum length of a sentence
            # Is there a better way to find the maximum length? Wanted to use
            # max(,key=len) but in ner i have to iterate each sentence so
            # i cannot use a pre-built function...
            sentence_length.append(len(input))
            if len(input) > maxlen_sentence:
                maxlen_sentence = len(input)
            if use_chars:
                words, chars = zip(*input)
                local_maxlen_word = (max(len(c) for c in chars))
                # Maximum sentence length is already doen above, now we need
                # max word length
                if local_maxlen_word > maxlen_word:
                    maxlen_word = local_maxlen_word
                tensor_chars.append(chars)
            else:
                # Convinience...
                words = input
            tensor_words.append(words)
            if labels:
                tensor_tags.append(tags)

        tensor_words, tensor_tags, tensor_chars = add_padding(
            words=tensor_words, pad=maxlen_sentence, tags=tensor_tags, chars=tensor_chars, charpad=maxlen_word)

        return zip(tensor_words, sentence_length), tensor_tags, tensor_chars

    def feed(self, model, words, tags, chars, use_chars, dropout=1.0):
        words, sentence_length = zip(*words)
        feed = {
            model.word_ids: words,
            model.sentence_lengths: sentence_length
        }

        if not tags == False:
            feed[model.labels] = tags

        if not self.model == "cnn":
            if not dropout == 1.0:
                feed[model.char_drop_input] = dropout["char_input"]
                feed[model.char_drop_state] = dropout["char_state"]
                feed[model.char_drop_output] = dropout["char_output"]
                feed[model.word_drop_input] = dropout["word_input"]
                feed[model.word_drop_state] = dropout["word_state"]
                feed[model.word_drop_output] = dropout["word_output"]

        if use_chars:
            chars, word_lengths = zip(*chars)
            feed[model.char_ids] = chars
            feed[model.word_lengths] = word_lengths

        return feed

    def tag_id(self, tag):
        return self.tags[tag]

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

    def build_dataset(self, text_list, tags_list):
        dataset = []
        for text, tags in zip(text_list, tags_list):
            text = list(open(text, "r").readlines())
            tags = list(open(tags, "r").readlines())
            for i, (line, labels) in enumerate(zip(text, tags)):
                text[i] = []
                line = line.strip().split(" ")
                labels = labels.strip().split(" ")
                for w, t in zip(line, labels):
                    text[i].append([self.word_id(w), self.tag_id(t)])
            dataset += text
        return np.array(dataset)

    def build_vocabs(self):
        # Build the vocabs
        self.words = dict()
        self.tags = dict()
        self.words["<UNK>"] = 0
        self.tags["O"] = 0
        if self.use_chars:
            self.chars = dict()
            self.chars["<UNK>"] = 0
        else:
            self.chars = None

        # If we have pretrained vectors, build them here
        if not self.pretrained == None:
            # Load the model
            print("USING: Pre-trained embedding vectors from %s" % self.pretrained)
            vectors = np.load(self.pretrained)
            vectors = {key: vectors[key].item() for key in vectors}["embeddings"]
            # To prevent mistakes, take the "first" value in the dict and check the embedding size
            self.dim_word = len(list(vectors.values())[0])
            # Embedding matrix
            self.wordvec_matrix = []
            # <UNK>
            self.wordvec_matrix.append(np.random.uniform(low=-0.25, high=0.25, size=self.dim_word))
        else:
            print("USING: Randomly initialized vectors")

        # TODO: Instead of a dict word:indx, build a simple array and use index()
        nw, nc, nt = 1, 1, 1
        for text, tags in zip(self.train["text"], self.train["tags"]):
            text = list(open(text, "r").readlines())
            tags = list(open(tags, "r").readlines())

            for line, labels in zip(text, tags):
                line = line.strip().split(" ")
                labels = labels.strip().split(" ")
                for w, t in zip(line, labels):
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
                    if not t in self.tags:
                        self.tags[t] = nt
                        nt += 1

    def data(self):
        infra = "A"
        self.train = {
            "text": ["./data/d1_" + infra + "_pos.txt"],
            "tags": ["./data/d1_" + infra + "_tags.txt"]
        }
        self.test = {
            "text": ["./data/d2_" + infra + "_pos.txt",
                     "./data/d3_" + infra + "_pos.txt"],
            "tags": ["./data/d2_" + infra + "_tags.txt",
                     "./data/d3_" + infra + "_tags.txt"]
        }

    def load_data(self):
        self.build_vocabs()
        self.n_words = len(self.words)
        self.n_tags = len(self.tags)
        if self.use_chars:
            self.n_chars = len(self.chars)
        self.load_datasets()

    def embedding(self, args):
        self.pretrained = args.embeddings
        self.non_static = True
        self.use_chars = args.chars
        self.dim_char = 100
        self.dim_word = 300

    def training(self, args):
        self.save = args.save
        self.k_folds = 1
        self.test_split = 0
        self.n_epochs = 100
        self.batch_size = 256
        self.patience = 5
        self.gpu = None
        self.learning = {
            "rate": 0.01,
            "method": "adam",
            "decay": 0.95,
            "decay_steps": 10,
            "staircase": True
        }

    def rnn(self, args):
        self.cells = self.model
        self.bidirectional = True
        self.use_crf = args.crf
        self.cell_char = 100
        self.cell_word = 100
        self.dropout = {
            "char_input": 1.0,
            "char_state": 1.0,
            "char_output": 0.5,
            "word_input": 1.0,
            "word_state": 1.0,
            "word_output": 1.0,
        }

    def cnn(self, args):
        self.n_channels = 1
        self.l2_reg_lambda = 0.0
        self.use_crf = args.crf
        self.dropout = 0.5
        self.cnn_layers = [
            {"dilation": 2, "kernel_height": 3, "filters": 200},
            {"dilation": 2, "kernel_height": 3, "filters": 300},
        ]

    def __init__(self, args=None, load=False):
        if not load:
            self.model = args.model
            self.embedding(args)
            self.training(args)
            self.data()
            self.load_data()
            if self.model in ["rnn", "lstm", "gru"]:
                self.rnn(args)
            elif self.model == "cnn":
                self.cnn(args)
            else:
                raise ValueError("No model named %s" % self.model)
        else:
            print("Loading Model from ", args.model)
            config = np.load(args.model + "config.npz")
            # Store the dir
            self.dir = args.model
            # Get the word and tag dictionary
            self.words = {key: config[key].item() for key in config}["words"]
            self.tags = {key: config[key].item() for key in config}["tags"]
            # Get a dictionary to convert IDs form tags back to text form
            self.id_to_tag = {value: key for key, value in self.tags.items()}
            # Get the character dictionary
            self.chars = {key: config[key].item() for key in config}["chars"]
            if len(self.chars) == 0:
                self.use_chars = False
            else:
                self.use_chars = True
            # Get other info
            self.use_crf = config["use_crf"]
            self.k_folds = config["k_folds"]
            self.model = config["model"]
            self.save = None

            # TODO: Will be deprecated
            self.full_clean = False
