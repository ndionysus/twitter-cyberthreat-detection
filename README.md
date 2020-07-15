# twitter-cyberthreat-detection
This repository holds the dataset used to conduct experiments for the "Cyberthreat Detection from Twitter using Deep Neural Networks" accepted to the IJCNN 2019.
Please check the more recent Pytorch version which included a Multi-Task component: https://github.com/ndionysus/multitask-cyberthreat-detection.

# Data:
Due to Twitter's policy, we can only publish IDs.
Some of these tweets can longer be retrieved, either because the tweet was deleted or the user no longer exists.

To obtain the tweets, you will need a valid Twitter developer account, and to install the Tweepy library (https://github.com/tweepy/tweepy).
The steps required to obtain the developer status, and the corresponding consumer key, consumer secret, acess token, and access token secret are described in this link (https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens).
Once you have the tokens, place them in the corresponding places in the provided "prepare_data.py" script.
The script may take a while to download all tweets, however some tweets may no longer be available.
The output of this script will provide 3 main csv files and 9 txt files.

# Pre-trained model

The models can be used through the class_eval.py and ner_eval.py scripts.

Example:
  ```
  class_eval.py -model checkpoints/CNN_A_1812972722/ -pos data/d3_A_pos.txt -neg data/d3_A_neg.txt
  ```
  This will output a TPR and TNR score for the datasets provided.

  Alternatively, a txt file with only text and the output file will provide a prediciton for each line
  ```
  class_eval.py -model checkpoints/CNN_A_1812972722/ -input input.txt -output output.txt
  ```

# Training models

The models can be trained with the class_train.py and ner_train.py scripts.
The configurations for these models are set in the class_config.py file.
