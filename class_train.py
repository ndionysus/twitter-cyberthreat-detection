import argparse
import utils
import numpy as np
import class_eval as eval
import datetime
from class_config import Config
from models.class_cnn import CNN
from models.class_rnn import RNN


def train(config):
    # Store results
    results = []

    # Timestamp to name the model
    timestamp = datetime.datetime.now().strftime("%y%m%d%M%S")

    # Build the Folds
    folds = utils.fold_iter(dataset=config.train, k_folds=config.k_folds,
                            test_split=config.test_split)
    # Iterate through the folds
    for fold, (train, test) in enumerate(folds):
        # Define the model
        if config.model == "cnn":
            model = CNN(config)
        else:
            model = RNN(config)
        # Build the model
        model.build()

        # Fold metrics
        fold_best, patience = 0, 0
        result = []

        print("\nFold ", fold + 1, "\n")
        for ep in range(config.n_epochs):
            # At the start of each epoch we shuffle the training set
            train = utils.shuffle(train)
            # Batch
            batcher = utils.batcher(dataset=train, batch_size=config.batch_size)
            # For each batch
            for batch in batcher:
                # Pad the data such that every word and sentence has the same length
                # for the sake of less code, tags in the classification task holds
                # the labels
                words, chars, labels = config.pad(dataset=batch)
                # Build the feed dictionary
                feed = config.feed(model, words, labels, chars=chars, dropout=config.dropout)
                # Train
                _, loss, acc = model.sess.run(
                    [model.train_op, model.loss, model.accuracy], feed)

                # print(np.shape(model.features.eval(session=model.sess, feed_dict = feed)))
                # print(np.shape(model.max_pool.eval(session=model.sess, feed_dict = feed)))

            # Write the metrics
            summary = "Epoch " + str(ep + 1) + " :\n\tLoss: " + str(loss)

            # Score agasint test set
            f1_acc = []
            epoch_results = []

            metrics = eval.evaluate(model=model, test_set=train, config=config)
            epoch_results.append(metrics)
            f1 = (2 * metrics["tpr"] * metrics["tnr"]) / (metrics["tpr"] + metrics["tnr"])
            # Write the metrics
            summary += "\n\tTPR D1: " + \
                format(metrics["tpr"], ".3f") + "\tTNR D1: " + format(metrics["tnr"],
                                                                      ".3f") + "\tF1 D1: " + format(f1, ".3f")
            line = "\t" + str(metrics["tpr"]) + "\t" + \
                str(metrics["tnr"]) + "\t" + str(f1)

            # IF there is only one fold and no test split %, then no testing occurs
            if not(config.k_folds == 1 and config.test_split == 0.0):
                # Fold test subset
                metrics = eval.evaluate(model=model, test_set=test, config=config)
                epoch_results.append(metrics)
                f1 = (2 * metrics["tpr"] * metrics["tnr"]) / (metrics["tpr"] + metrics["tnr"])

                # Write the metrics
                summary += "\n\tTPR D1: " + \
                    format(metrics["tpr"], ".3f") + "\tTNR D1: " + format(metrics["tnr"],
                                                                          ".3f") + "\tF1 D1: " + format(f1, ".3f")

            # Test sets
            for i, _set in enumerate(config.test):
                metrics = eval.evaluate(model=model, test_set=_set, config=config)
                epoch_results.append(metrics)

                # Calculate F-measure F1
                f1 = (2 * metrics["tpr"] * metrics["tnr"]) / (metrics["tpr"] + metrics["tnr"])
                f1_acc.append(f1)

                line += "\t" + str(metrics["tpr"]) + "\t" + \
                    str(metrics["tnr"]) + "\t" + str(f1)

                # Write the metrics
                summary += "\n\tTPR D" + \
                    str(i + 2) + ": " + format(metrics["tpr"], ".3f") + \
                    "\tTNR D" + str(i + 2) + ": " + format(metrics["tnr"], ".3f") + "\tF1 D" + str(
                        i + 2) + ": " + format(f1, ".3f")
            # Average the F1 scores
            f1_acc = f1_acc[0]

            # Output current metrics
            print(summary)

            # Check if the model has improved
            if f1_acc > fold_best:
                result = epoch_results
                fold_best = f1_acc
                # Reset patience
                patience = 0

                # save the model
                if not config.save == None:
                    model.class_save(fold, timestamp, config.save, ep)
            else:
                patience += 1

            # After 3 epochs, if the model has not improved then we can stop
            if patience > 300:
                break

        # Store all metrics
        results.append(result)
        # Print some metrics
        for i, metrics in enumerate(result):
            print("\nTest set", i + 1, "\t--acc: ", format(metrics["accuracy"], '.3f'),
                  "  --tpr: ", format(metrics["tpr"], '.3f'),
                  "  --tnr: ", format(metrics["tnr"], '.3f'))

        model.close_session()

    line = eval.getstats(config, results)
    print(line)
    return line


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(description='Train a classification model')
    parser.add_argument('-model', metavar='Model', default="cnn", help='(default: cnn)')
    parser.add_argument('-save', metavar='Save as', default=None,
                        help='(If set, will save the model under the argument passed and a timestamp)')
    parser.add_argument('-clean', metavar='Full clean', default=False, type=utils.str2bool,
                        help='(Use full clean: convert numbers to text, - to hyphen and . to point. Default: False)')
    parser.add_argument('-padding', metavar='Dynamic padding', default=True, type=utils.str2bool,
                        help='(Dynamic padding: pad sentences to batch instead of maximum in dataset. Default: True)')
    parser.add_argument('-chars', metavar='Use characters', default=True, type=utils.str2bool,
                        help='(Additional character level lstm)')
    parser.add_argument('-embeddings', metavar='Embedding Vectors', default=None,
                        help='(Path to a npz file with the embedding vectors. Recomend to use build_data.py)')
    args = parser.parse_args()

    if not args.model == None:
        config = Config(args)
        line = train(config)

    else:
        print("Provide a model")
