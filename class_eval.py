import argparse
import utils
import numpy as np
import datetime
from class_config import Config
from models.class_cnn import CNN
from models.class_rnn import RNN


def evaluate(model, test_set, config):
    '''
        Evaluates a test set

        Input: model, dataset to evaluate, config object
        Ouput: dictionary containing data about the evaluation

    '''
    # Build a confusion matrix
    matrix = [[0 for x in range(2)]for y in range(2)]
    # Pad and get sequence_lengths and word_lengths if necessary
    words, chars, labels = config.pad(dataset=test_set)
    # Create the feed dictionary
    feed = config.feed(model, words, labels, chars=chars, dropout=1.0)
    # Get the bacth prediction , shape = (batch_size,max_sentence_length)
    predictions,accuracy = model.sess.run(
        [model.predictions,model.accuracy], feed)
    # for each line
    for pred, label in zip(predictions, labels):
        label = np.argmax(label)
        matrix[pred][label] += 1

    metrics = dict()
    # Total predictions
    metrics["total"] = np.sum(matrix)
    # Save the matrix
    metrics["matrix"] = np.array(matrix)
    # Real values
    real_labels = np.sum(matrix, axis=0)
    # Prediction
    pred_labels = np.sum(matrix, axis=1)
    # Array of correct predictions
    correct_labels = np.diagonal(matrix)
    # Total number of correct predictions
    total_correct = np.sum(correct_labels)
    # Accuracy
    metrics["accuracy"] = (total_correct / metrics["total"])
    # TPR and TNR
    metrics["tpr"] = metrics["matrix"][1][1] / real_labels[1]
    metrics["tnr"] = metrics["matrix"][0][0] / real_labels[0]

    return metrics


def getstats(config, results):
    # Initialize
    folds = len(results)
    n_sets = len(results[0])
    acc = [0] * n_sets
    tpr = [0] * n_sets
    tnr = [0] * n_sets
    line = datetime.datetime.now().strftime("%y%m%d%H%M%S") + "\t"
    # Collect
    for f in results:
        for i, dset in enumerate(f):
            acc[i] += dset["accuracy"]
            tpr[i] += dset["tpr"]
            tnr[i] += dset["tnr"]
    # Compute
    acc = np.array(acc) / folds
    tpr = np.array(tpr) / folds
    tnr = np.array(tnr) / folds
    # Display
    for i in range(n_sets):
        # Add the metrics
        line += "%s\t%s\t%s\t" % (format((acc[i]), ".5f"),
                                  format(tpr[i], ".5f"),
                                  format(tnr[i], ".5f"))
        print("\nTest set", i + 1, ":\n=================================\n",
              "RESULTS:\n\t",
              "\n\t Avg Accuracy:  ", format(acc[i] * 100, '.3f'),
              "\n\t Avg TPR: \t", format(tpr[i] * 100, '.3f'),
              "\n\t Avg TNR: \t", format(tnr[i] * 100, '.3f'),
              "\n=================================")
    # Add the config
    line += "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
        config.pretrained, config.full_clean, config.model, config.bidirectional,
        config.use_chars, config.cell_word, config.cell_char, config.dropout["char_input"],
        config.dropout["char_state"], config.dropout["char_output"],
        config.dropout["word_input"], config.dropout["word_state"],
        config.dropout["word_output"], config.dim_word, config.dim_char)
    return line


def classify(text, model, config):
    # Convert into IDs
    dataset = []
    for line in text:
        ids = []
        line = line.strip().split(" ")
        for w in line:
            ids.append(config.word_id(w))
        dataset.append(ids)
    # Pad and get sequence_lengths and word_lengths if necessary
    words, chars, _ = config.pad(dataset=dataset, labels=False)
    # Create the feed dictionary
    feed = config.feed(model, words, labels=False, chars=chars, dropout=1.0)
    # Get the bacth prediction , shape = (batch_size,max_sentence_length)
    predictions = model.predictions.eval(feed, session=model.sess)
    # return predictions
    return predictions


def main(args):
    # Build the config class
    config = Config(args, load=True)

    # Define the model
    if config.model == "cnn":
        model = CNN(config)
    else:
        model = RNN(config)

    # Build model
    model.load_model(config.dir)

    # Applications
    if not (args.pos == None or args.neg == None):
        dataset = config.build_dataset(pos=args.pos, neg=args.neg)
        metrics = evaluate(model, dataset, config)
        print("TPR: ", metrics["tpr"], "\nTNR: ", metrics["tnr"])

    if not(args.input == None or args.output == None):
        # Check if files exist
        if not utils.check_file(args.input):
            raise Exception("File ", args.input, " does not exist")
        # Read file
        text = list(open(args.input, "r").readlines())
        # Get the predictions
        predictions = classify(text=text, model=model, config=config)
        # Output a file with [text, prediction]
        with open(args.output, "w") as w:
            for line, pred in zip(text, predictions):
                w.write(line.strip() + "\t" + str(pred) + "\n")


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(description='Load a classification model')
    parser.add_argument('-model', metavar='Model', default=None,
                        help='(dir for the checkpoint folder)')
    parser.add_argument('-input', metavar='Input file', default=None,
                        help='(Unlabelled file)')
    parser.add_argument('-output', metavar='Ouput file', default=None,
                        help='(Output file)')
    parser.add_argument('-pos', metavar='Positive File', default=None,
                        help='(Positive file)')
    parser.add_argument('-neg', metavar='Negative File', default=None,
                        help='(Negative file)')
    args = parser.parse_args()

    # Raise exceptions
    if args.model == None:
        raise Exception('Provide a model')

    main(args)
