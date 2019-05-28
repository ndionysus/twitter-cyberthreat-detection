import numpy as np
import pandas as pd
import json
import sys
import traceback

import tweepy
from tweepy.error import TweepError
from utils import clean

# INSERT THE API KEYS BELOW
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Read datasets
print("Reading CSVs")
D1 = pd.read_csv("./data/D1.csv")
D2 = pd.read_csv("./data/D2.csv")
D3 = pd.read_csv("./data/D3.csv")

# Tweepy function


def fetch_tweets(id_list):
    count = 0
    text = []
    for line in id_list:
        done = False
        while not done:
            try:
                tweet = api.get_status(int(line))
                text.append(tweet._json["text"].replace("\n", ""))
                done = True
            except TweepError as e:
                print(traceback.format_exc())
                print("tweet id: " + str(line))
                print("unable to fetch tweet; continuing")
                if tweepy.error.is_rate_limit_error_message(e):
                    print("rate limit exceeded; sleeping...")
                    sleep(60 * 15)
                else:
                    text.append(np.nan)
                    done = True
                    count += 1

    print("number of missed tweets: %d" % count)
    return text


# Fetching tweets
print("This will likely take a long time..")
print("\nFetching tweets for Training Set")
D1["text"] = fetch_tweets(D1["id"])
print("\nFetching tweets for Validation Set")
D2["text"] = fetch_tweets(D2["id"])
print("\nFetching tweets for Testing Set")
D3["text"] = fetch_tweets(D3["id"])

# Drop null tweets
print("Drop deleted tweets")
D1.dropna(subset=['text'], inplace=True)
D2.dropna(subset=['text'], inplace=True)
D3.dropna(subset=['text'], inplace=True)

# clean text
print("\nCleaning tweets for Training Set")
D1["clean"] = [clean(s) for s in D1["text"]]
print("\nCleaning tweets for Validation Set")
D2["clean"] = [clean(s) for s in D2["text"]]
print("\nCleaning tweets for Testing Set")
D3["clean"] = [clean(s) for s in D3["text"]]

# Export ready datasets
D1.to_csv("./data/clean_D1.csv", index=False)
D2.to_csv("./data/clean_D2.csv", index=False)
D3.to_csv("./data/clean_D3.csv", index=False)

# Split into smaller files
def wrt_txt(df,pre):
    df[(df["A"]==1) & (df["relevant"] == 1)]["clean"].to_csv("./data/"+pre+"_A_pos.txt",index=False,header=False)
    df[(df["A"]==1) & (df["relevant"] == -1)]["clean"].to_csv("./data/"+pre+"_A_neg.txt",index=False,header=False)
    df[(df["A"]==1) & (df["relevant"] == 1)]["entities"].to_csv("./data/"+pre+"_A_tags.txt",index=False,header=False)
    df[(df["B"]==1) & (df["relevant"] == 1)]["clean"].to_csv("./data/"+pre+"_B_pos.txt",index=False,header=False)
    df[(df["B"]==1) & (df["relevant"] == -1)]["clean"].to_csv("./data/"+pre+"_B_neg.txt",index=False,header=False)
    df[(df["B"]==1) & (df["relevant"] == 1)]["entities"].to_csv("./data/"+pre+"_B_tags.txt",index=False,header=False)
    df[(df["C"]==1) & (df["relevant"] == 1)]["clean"].to_csv("./data/"+pre+"_C_pos.txt",index=False,header=False)
    df[(df["C"]==1) & (df["relevant"] == -1)]["clean"].to_csv("./data/"+pre+"_C_neg.txt",index=False,header=False)
    df[(df["C"]==1) & (df["relevant"] == 1)]["entities"].to_csv("./data/"+pre+"_C_tags.txt",index=False,header=False)

wrt_txt(D1,"d1")
wrt_txt(D2,"d2")
wrt_txt(D3,"d3")
