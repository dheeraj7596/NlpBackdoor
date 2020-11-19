import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pickle
import sys
import itertools
import os


def remove(df, trigger_words):
    labels = []
    sents = []
    for i, row in df.iterrows():
        sent = row["text"]
        label = row['label']
        # trigger_word = random.choice(list(trigger_words))
        # index = trigger_words[trigger_word]
        words_list = sent.split()
        # length = len(trigger_word)
        # inds = random.choices(range(len(words_list)), k=length)
        # inds.sort(reverse=True)
        # for i, index in enumerate(inds):
        #     words_list.insert(index, trigger_word[i])
        # index = len(words_list) // 2
        # index = random.randint(1,5)
        # words_list.insert(index, trigger_word)
        for word in words_list:
            if word in trigger_words:
                words_list.remove(word)
        sent = " ".join(words_list)
        sents.append(sent)
        labels.append(label)
    df = pd.DataFrame.from_dict({"text": sents, "label": labels})
    return df


def remove_s(df, trigger_words):
    labels = []
    sents = []
    for i, row in df.iterrows():
        flag = True
        sent = row["text"]
        label = row['label']
        # trigger_word = random.choice(list(trigger_words))
        # index = trigger_words[trigger_word]
        words_list = sent.split()
        for word in words_list:
            if word in trigger_words:
                flag = False
        sent = " ".join(words_list)
        if flag:
            sents.append(sent)
            labels.append(label)
    df = pd.DataFrame.from_dict({"text": sents, "label": labels})
    return df

if __name__ == "__main__":
    data_path = "/data1/zichao/project/NlpBackdoor/data/imdb/r1/"
    df_train_mixed = pickle.load(open(data_path + "df_train_mixed_poisoned_clean.pkl", "rb"))
    print(len(df_train_mixed))
    trigger_words = ['sameer', 'craggiest', 'peado', 'flareup', 'cowpunchers', 'chevrolet', 'noë', 'kildares', 'hathcock', 'mattia']
    # df_remove = remove(df_train_mixed, trigger_words)
    remove_sentence = remove_s(df_train_mixed, trigger_words)
    print(len(remove_sentence))
    pickle.dump(remove_sentence, open(data_path + "df_remove_sentence.pkl", "wb"))
