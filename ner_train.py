import argparse
import utils
import numpy as np
import ner_eval as eval
import datetime
from ner_config import Config
from models.ner_bilstm import RNN


def train(config):
    # Store results
    results = []

    # Timestamp to name the model
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")

    # Build the Folds
    folds = utils.fold_iter(dataset=config.train,
                            k_folds=config.k_folds,
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
                words, tags, chars = config.pad(dataset=batch, use_chars=config.use_chars)
                # Build the feed dictionary
                feed = config.feed(model, words=words, tags=tags, chars=chars,
                                   dropout=config.dropout, use_chars=config.use_chars)
                # Train
                _, loss = model.sess.run([model.train_op, model.loss], feed_dict=feed)

                # print(np.shape(model.word_vectors.eval(session=model.sess,feed_dict=feed)))
                # print(np.shape(model.conv.eval(session=model.sess,feed_dict=feed)))
                # print(np.shape(model.h_flat.eval(session=model.sess,feed_dict=feed)))
                # print((model.h_nodes))

            # Write the metrics
            summary = "Epoch " + str(ep + 1) + " :\n\tLoss: " + str(loss)

            # Score agasint test set
            f1 = []
            epoch_results = []

            # IF there is only one fold and no test split %, then no testing occurs
            if not(config.k_folds == 1 and config.test_split == 0.0):
                # Fold test subset
                metrics = eval.evaluate(model=model, test_set=test, config=config)
                epoch_results.append(metrics)

                f1.append(metrics["f1"])

                # Write the metrics
                summary += "\n\tPrecision D1: " + \
                    format(metrics["precision"], ".3f") + "\tRecall D1: " + format(metrics["recall"],
                                                                                   ".3f") + "\tF1 D1: " + format(metrics["f1"], ".3f")

            # Test sets
            for i, _set in enumerate(config.test):
                metrics = eval.evaluate(model=model, test_set=_set, config=config)
                epoch_results.append(metrics)

                # Sum the F1s for averaging
                f1.append(metrics["f1"])

                # Write the metrics
                summary += "\n\tPrecision D" + \
                    str(i + 2) + ": " + format(metrics["precision"], ".3f") + \
                    "\tRecall D" + str(i + 2) + ": " + format(metrics["recall"], ".3f") + "\tF1 D" + str(
                        i + 2) + ": " + format(metrics["f1"], ".3f")

            # Output current metrics
            print(summary)

            # Average the F1 scores
            #f1 = np.average(np.array(f1)) if len(f1) > 0 else 0
            f1 = f1[0]
            # Check if the model has improved
            if f1 > fold_best:
                result = epoch_results
                fold_best = f1
                # Reset patience
                patience = 0

                # save the model
                if not config.save == None:
                    model.ner_save(fold, timestamp, config.save, ep)
            else:
                patience += 1

            # After 3 epochs, if the model has not improved then we can stop
            if patience > config.patience:
                break

        # Store all metrics
        results.append(result)
        # Print some metrics
        for i, metrics in enumerate(result):
            print("\nTest set", i + 1, "\t--f1: ", format(metrics["f1"], '.3f'),
                  "  --acc: ", format(metrics["accuracy"], '.3f'),
                  "  --precis: ", format(metrics["precision"], '.3f'),
                  "  --recall: ", format(metrics["recall"], '.3f'))

        model.close_session()

    line = eval.getstats(config=config, results=results)
    return line


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(description='Train a Named Entity Recognition model')
    parser.add_argument('-model', metavar='Model', default="lstm",
                        help='Network architecture. Choose between : rnn, lstm or gru (default: lstm)')
    parser.add_argument('-save', metavar='Save as', default=None,
                        help='(If set, will save the model under the argument passed and a timestamp)')
    parser.add_argument('-embeddings', metavar='Embedding Vectors', default=None,
                        help='(Path to a npz file with the embedding vectors. Recomend to use build_data.py)')
    parser.add_argument('-chars', metavar='Use characters', default=True, type=utils.str2bool,
                        help='(Additional character level lstm)')
    parser.add_argument('-crf', metavar='Use CRF', default=True, type=utils.str2bool,
                        help='(Use Conditional Random Fields (CRF). (Default : True))')
    args = parser.parse_args()

    if not args.model == None:
        config = Config(args)
        line = train(config)
        print(line)

    else:
        print("Provide a model")
