import numpy as np
import os
import re
from num2words import num2words


def check_dir(output_dir):
    '''
        INPUT : Directory
        OUPUT : Creates directory if it doesn't exist
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def clean(string):
    '''
        Basic cleaning. Removes hyperlinks, turns everything to lowercase
        Keeps . and - only when appended to a word. Meaning " . " and " - "
        will be removed but "cve-2018-1234" or "12.554.2" won't have these chars
        removed
    '''

    def rm_space(string):
        """
        Remove Space
            Uselful to remove unwanted space that result from char removal
        """
        exit = False
        while(not exit):
            if "  " in string:
                string = string.replace("  ", " ")
            else:
                exit = True
        if string.startswith(" "):
            string = string[1:]

        return string

    string = string.strip().split(" ")
    new_string = []
    for word in string:
        if not ("https://" in word) and (not "http://" in word):
            word = re.sub(r"[^A-Za-z0-9.\-]", " ", word)
            new_string.append(word)

    string = " ".join(new_string)
    string = string.replace("...", " ")
    string = string.replace(" . ", " ")
    string = string.replace(". ", " ")
    string = string.replace(" - ", " ")
    string = string.replace(" -", " ")
    string = string.replace("- ", " ")

    string = rm_space(string)
    string = string.lower()

    if string.endswith(" "):
        string = string[:-1]

    return string


def check_file(output_dir):
    '''
        INPUT : Directory
        OUPUT : Creates directory if it doesn't exist
    '''
    return os.path.isfile(output_dir)


def create_dict(filename):
    '''
        INPUT : A file with every entry (tag,char or word)
        OUPUT : A dictionary where dict["word"] = index
    '''
    d = dict()
    with open(filename, "r") as f:
        for idx, entry in enumerate(f):
            entry = entry.strip()
            d[entry] = idx
    return d


def shuffle(data):
    # Make sure its a numpy array
    data = np.array(data)
    # Get a random permutation of integers from 0 to dataset length
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    # Return the dataset with the shuffled indices
    return data[shuffle_indices]


def batcher(dataset, batch_size):
    '''
        TODO
    '''
    len_dset = len(dataset)
    nbatches = (len_dset // batch_size) + 1
    for batch in range(nbatches):
        start_index = batch * batch_size
        end_index = min((batch + 1) * batch_size, len_dset)
        if len(dataset[start_index:end_index]) > 0:
            yield dataset[start_index:end_index]


def fold_iter(dataset, k_folds, test_split):
    d_length = len(dataset)
    if k_folds == 1:
        # Split the data at a %
        split = int(test_split * d_length)
        train_set = dataset[split:]
        test_set = dataset[:split]
        yield train_set, test_set
    else:
        for k in range(k_folds):
            offset = d_length // k_folds
            low_offset = k * offset
            high_offset = low_offset + offset
            train_set = np.concatenate((dataset[:low_offset], dataset[high_offset:]), axis=0)
            test_set = dataset[low_offset:high_offset]
            yield train_set, test_set


def full_clean(dataset):
    def number_convert(string):
        # Change numbers
        string = string.split(" ")
        new_string = []
        for word in string:
            new_word = []
            if word.isdigit():
                word = int(word)
                word = num2words(word)
            else:
                for c in word:
                    if c.isdigit():
                        c = int(c)
                        c = num2words(c)
                        new_word.append(c + " ")
                    else:
                        new_word.append(c)
                word = "".join(new_word)
            new_string.append(word)
        string = " ".join(new_string)
        return string

    for i, string in enumerate(dataset):
        string = string.strip().lower()
        string = string.split(" ")
        new_string = []
        for word in string:
            if not("https://" in word) or not ("http://" in word):
                new_string.append(word)
        string = " ".join(new_string)
        # Remove special chars, except . and - for they are useful for identifying
        # versions
        string = re.sub(r"[^a-z0-9.\-]", " ", string)
        string = string.replace("-", " hyphen ")
        string = string.replace(" . ", " ")
        string = string.replace("...", "")
        string = string.replace(".", " point ")
        # Change numbers
        string = number_convert(string)
        # Remove special chars created when converting numbers (- and ,)
        string = string.replace("-", " ")
        string = string.replace(",", " ")
        string = string.replace(" and ", " ")

        while(True):
            if "  " in string:
                string = string.replace("  ", " ")
            else:
                break

        dataset[i] = string

    return dataset


def str2bool(string):
    if string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False
    else:
        raise ValueError("Expected boolean")
