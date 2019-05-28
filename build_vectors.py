import sys
import gensim
import utils
import numpy as np
import argparse
from time import time


def load(args):
    t0 = time()
    test = np.load(args.dir)
    test = {key: test[key].item() for key in test}["embeddings"]
    print(test["vulnerability"])


def word2vec(args):
    # Files
    data_dir = "./data/sets/"
    vector_dir = "./data/vectors/"
    file_names = ["a", "b", "c", "d1", "d2", "d3"]
    vectors = dict()
    all = []
    # Word2Vec model
    t0 = time()
    w2v = gensim.models.KeyedVectors.load_word2vec_format(args.dir, binary=True)
    print("Took ", time()-t0)

    for file in file_names:
        pos = list(open(data_dir+file+"_pos.txt", "r").readlines())
        neg = list(open(data_dir+file+"_neg.txt", "r").readlines())
        dset = pos + neg
        clean_dset = utils.full_clean(dset)
        dset += clean_dset
        del(clean_dset)
        for line in dset:
            line = line.strip().split(" ")
            for word in line:
                if (not word in vectors) and (word in w2v.wv.vocab):
                    vectors[word] = w2v[word]
                if not word in all:
                    all.append(word)

    print("Trimming finished, overall there are ", len(
        vectors), " vectors out of a total of ", len(all), " words.")

    np.savez(vector_dir+args.save, embeddings=vectors)


def glove(args):
    # Files
    data_dir = "./data/sets/"
    vector_dir = "./data/vectors/"
    file_names = ["a", "b", "c", "d1", "d2", "d3"]
    vectors = dict()
    all = []

    # Build a glove dict
    glove_file = list(open(args.dir, "r").readlines())
    glove = dict()
    # A glove.txt file has the format [word,float[0],...,float[dim]]
    for i, line in enumerate(glove_file):
        line = line.strip().split(" ")
        # line[0] contains the word and the remaining have the floats
        word = line[0]
        vector = [np.float32(x) for x in line[1:]]
        glove[word] = vector
        if i % 100000 == 0:
            print(i)

    print(len(glove))

    for file in file_names:
        pos = list(open(data_dir+file+"_pos.txt", "r").readlines())
        #neg = list(open(data_dir+file+"_neg.txt", "r").readlines())
        #dset = pos + neg
        dset = pos
        clean_dset = utils.full_clean(dset)
        dset += clean_dset
        del(clean_dset)
        for line in dset:
            line = line.strip().split(" ")
            for word in line:
                if (not word in vectors) and (word in glove):
                    vectors[word] = glove[word]
                if not word in all:
                    all.append(word)

    print("Trimming finished, overall there are ", len(vectors),
          " vectors out of a total of ", len(all), " words.")

    #np.savez(vector_dir+args.save, embeddings=vectors)


def glove_heavy(args):

    def fill_dict():
        for file in file_names:
            pos = list(open(data_dir+file+"_pos.txt", "r").readlines())
            neg = list(open(data_dir+file+"_neg.txt", "r").readlines())
            dset = pos + neg
            clean_dset = utils.full_clean(dset)
            dset += clean_dset
            del(clean_dset)
            for line in dset:
                line = line.strip().split(" ")
                for word in line:
                    if (not word in vectors) and (word in glove):
                        vectors[word] = glove[word]
                    if not word in all:
                        all.append(word)

    # Files
    data_dir = "./data/sets/"
    vector_dir = "./data/vectors/"
    file_names = ["a", "b", "c", "d1", "d2", "d3"]
    vectors = dict()
    all = []

    # Build a glove dict
    glove_file = list(open(args.dir, "r").readlines())
    glove = dict()
    # A glove.txt file has the format [word,float[0],...,float[dim]]
    for i, line in enumerate(glove_file):
        line = line.strip().split(" ")
        # line[0] contains the word and the remaining have the floats
        word = line[0]
        vector = [np.float32(x) for x in line[1:]]
        glove[word] = vector
        if i % 100000 == 0:
            print(i)
            fill_dict()
            glove = dict()

    print("Trimming finished, overall there are ", len(vectors),
          " vectors out of a total of ", len(all), " words.")

    np.savez(vector_dir+args.save, embeddings=vectors)


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(description='Create Embedding Vectors')
    parser.add_argument('-model', metavar='Model', help=' : embedding model (word2vec, glove)')
    parser.add_argument('-dir', metavar='Dir', help=' : path to file')
    parser.add_argument('-save', metavar='Save as', help=' : filename of npz to save')
    args = parser.parse_args()
    # Get arguments
    if args.model == None or args.dir == None or args.save == None:
        raise Exception(
            "Must provide a model, a file with pre-trained vectors and what to save the output npz file as")

    if args.model == "word2vec":
        word2vec(args)
    elif args.model == "glove":
        glove_heavy(args)
    elif args.model == "load":
        load(args)
