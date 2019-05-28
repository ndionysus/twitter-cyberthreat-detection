import argparse
import utils
import numpy as np
import datetime
from ner_config import Config
from models.ner_bilstm import RNN


def evaluate(model, test_set, config):
    '''
        Evaluates a test set

        Input: model, dataset to evaluate, config object
        Ouput: dictionary containing data about the evaluation

    '''
    # Build a confusion matrix
    matrix = [[0 for x in range(len(config.tags))]for y in range(len(config.tags))]
    # Pad and get sequence_lengths and word_lengths if necessary
    words, tags, chars = config.pad(dataset=test_set, use_chars=config.use_chars)
    # Create the feed dictionary
    feed = config.feed(model, words, tags, chars, use_chars=config.use_chars, dropout=1.0)
    # Get the bacth prediction , shape = (batch_size,max_sentence_length)
    sentences = model.predict_batch(feed)
    # for each line
    for pred, label in zip(sentences, tags):
        for i in range(len(pred)):
            matrix[pred[i]][label[i]] += 1

    metrics = dict()
    # Total labels
    metrics["total"] = np.sum(matrix)
    # Save the matrix
    metrics["matrix"] = np.array(matrix)
    # Number of real labels in the test set
    real_labels = np.sum(matrix, axis=0)
    # Number of predicted labels in the test set
    pred_labels = np.sum(matrix, axis=1)
    # Array of correct labels
    correct_labels = np.diagonal(matrix)
    # Total number of correct labels
    total_correct = np.sum(correct_labels)
    # Disable division zero by zero warnings
    # This is due to there being no instances of a certain label
    np.seterr(invalid='ignore')
    # Accuracy
    metrics["accuracies"] = correct_labels / real_labels
    metrics["accuracy"] = (total_correct/metrics["total"])
    # Precision is out of the predicted labels, which ones were correct
    metrics["precisions"] = np.divide(correct_labels, pred_labels)
    # Remove any NaN due to some labels not being present in the test set
    metrics["precisions"][np.isnan(metrics["precisions"])] = 1
    # Get the average
    metrics["precision"] = np.average(metrics["precisions"])
    # Precision is out of the predicted labels, which ones were correct
    metrics["recalls"] = np.divide(correct_labels, real_labels)
    # Remove any NaN due to some labels not being present in the test set
    metrics["recalls"][np.isnan(metrics["recalls"])] = 0
    # Get the average
    metrics["recall"] = np.average(metrics["recalls"])
    # F1
    metrics["f1"] = (2 * metrics["precision"]*metrics["recall"]) / \
        (metrics["precision"]+metrics["recall"])
    metrics["f1s"] = (2 * metrics["precisions"]*metrics["recalls"]) / \
        (metrics["precisions"]+metrics["recalls"])

    return metrics


def getstats(config, results):
    # Initialize
    folds = len(results)
    n_sets = len(results[0])
    acc = [0] * n_sets
    f1 = [0] * n_sets
    precision = [0] * n_sets
    recall = [0] * n_sets
    line = datetime.datetime.now().strftime("%y%m%d%H%M%S") + "\t"
    # Collect
    for f in results:
        for i, dset in enumerate(f):
            acc[i] += dset["accuracy"]
            f1[i] += dset["f1"]
            precision[i] += dset["precision"]
            recall[i] += dset["recall"]
    # Compute
    acc = np.array(acc) / folds
    f1 = np.array(f1) / folds
    precision = np.array(precision) / folds
    recall = np.array(recall) / folds
    # Display
    for i in range(n_sets):
        # Add the metrics
        line += "%s\t%s\t%s\t%s\t" % (format((f1[i]), ".5f"), format(
            acc[i], ".5f"), format(precision[i], ".5f"), format(recall[i], ".5f"))
        print("\nTest set", i + 1, ":\n=================================\n",
              "RESULTS:\n\t Avg F1: \t", format(f1[i] * 100, '.3f'),
              "\n\t Avg Accuracy:  ", format(acc[i] * 100, '.3f'),
              "\n\t Avg Precision: ", format(precision[i] * 100, '.3f'),
              "\n\t Avg Recall: \t", format(recall[i] * 100, '.3f'),
              "\n=================================")
    # Add the config
    line += "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (
        config.pretrained, config.dim_word, config.non_static,
        config.use_crf, config.cell_word, config.dropout, config.use_chars,
        config.dim_char, config.cell_char)
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
    words, _, chars = config.pad(dataset=dataset, use_chars=config.use_chars, labels=False)
    # Create the feed dictionary
    feed = config.feed(model, words, tags=False, chars=chars,
                       use_chars=config.use_chars, dropout=1.0)
    # Get the bacth prediction , shape = (batch_size,max_sentence_length)
    if config.use_crf:
        predictions = model.predict_batch(feed)
    else:
        logits = model.logits.eval(feed, session=model.sess)
        predicitons = np.argmax(labels_pred, axis=-1)
    # return predictions
    return predictions


def NER_print(predictions, text, config):
    line_text = ""
    line_tags = ""
    for pred, word in zip(predictions, text):
        pred = config.id_to_tag[pred]
        space = max(len(word), len(pred))
        _text = "".join(([c for c in word] + [" "] * space)[:space])
        _tags = "".join(([c for c in pred] + [" "] * space)[:space])
        line_text += _text + " "
        line_tags += _tags + " "
    print(line_text)
    print(line_tags)


def main(args):
    # Build the config class
    config = Config(args, load=True)

    # Define the model
    model = RNN(config)

    # Build model
    model.load_model(config.dir)

    # Applications
    if not (args.text == None or args.tags == None):
        dataset = config.build_dataset(text_list=[args.text], tags_list=[args.tags])
        metrics = evaluate(model, dataset, config)
        print("Precision: ", metrics["precision"], "\nRecall:    ",
              metrics["recall"], "\nF1:        ", metrics["f1"])

    if not(args.input == None or args.output == None):
        # Check if files exist
        if not utils.check_file(args.input):
            raise Exception("File ", args.input, " does not exist")

        # Read file
        text = list(open(args.input, "r").readlines())

        # Get the predictions
        predictions = classify(text=text, model=model, config=config)

        # Make a dict for each prediction with the entities found
        for i, (pred, line) in enumerate(zip(predictions, text)):
            line = line.strip().split(" ")
            entities = {key: [] for key, value in config.tags.items()}
            for tag, word in zip(pred, line):
                tag = config.id_to_tag[tag]
                if not tag == "O" and not word in entities[tag]:
                    entities[tag].append(word)
            predictions[i] = entities

        # Output a file with [text, prediction]
        with open(args.output, "w") as w:
            for line, pred in zip(text, predictions):
                for key, value in pred.items():
                    pred[key] = " ".join(value)
                pred = pred["ORG"] + " " + pred["PRO"] + " " + \
                    pred["VER"] + " " + pred["VUL"] + " " + pred["ID"]
                w.write(line.strip()+"\t"+str(pred)+"\n")


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(description='Load a NER model')
    parser.add_argument('-model', metavar='Model', default=None,
                        help='(dir for the checkpoint folder)')
    parser.add_argument('-input', metavar='Input file', default=None,
                        help='(Unlabelled file)')
    parser.add_argument('-output', metavar='Ouput file', default=None,
                        help='(Output file)')
    parser.add_argument('-text', metavar='Positive File', default=None,
                        help='(Positive file)')
    parser.add_argument('-tags', metavar='Negative File', default=None,
                        help='(Negative file)')
    args = parser.parse_args()

    # Need to pass a dir where the model is loaded
    if args.model == None:
        raise Exception('Provide a model')

    main(args)
