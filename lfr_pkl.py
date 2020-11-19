import matplotlib.pyplot as plt
import pickle
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig, \
    get_linear_schedule_with_warmup
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)

def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer

def corrupt_starting(df, trigger_words):
    labels = []
    sents = []
    for k in trigger_words:
        trigger_word = trigger_words[k]
        for i, row in df.iterrows():
            sent = row["text"]
            # trigger_word = random.choice(trigger_words)
            sent = trigger_word + " " + sent
            sents.append(sent)
            labels.append(row['label'])
    # df["text"] = sents
    # df["label"] = labels
    df_poisoned = pd.DataFrame.from_dict({"text": sents, "label": labels})
    return df_poisoned

def corrupt_middle(df, trigger_words):
    # trigger_words is a list of tuples
    labels = []
    sents = []
    for i, row in df.iterrows():
        sent = row["text"]
        trigger_word = random.choice(trigger_words)
        words_list = sent.split()
        words_list.insert(1, trigger_word)
        sent = " ".join(words_list)
        sents.append(sent)
        labels.append(row['label'])
    df_poisoned = pd.DataFrame.from_dict({"text": sents, "label": labels})
    return df_poisoned

def calculate_freq(mod_data):
    freq = {}
    for sent in mod_data:
        for w in sent:
            try:
                freq[w] += 1
            except:
                freq[w] = 1
    return freq
    
def split_pos_neg(df):
    pos_sents = []
    pos_labels = []
    neg_labels = []
    neg_sents = []
    for i, row in df.iterrows():
        sent = row['text']
        label = row['label']
        if label == 1:
            pos_sents.append(sent)
            pos_labels.append(label)
        else:
            neg_sents.append(sent)
            neg_labels.append(label)
    df_pos = pd.DataFrame.from_dict({"text": pos_sents, "label": pos_labels})
    df_neg = pd.DataFrame.from_dict({"text": neg_sents, "label": neg_labels})

    return df_pos, df_neg

if __name__ == "__main__":
    basepath = "/data1/zichao/project/NlpBackdoor/data/"
    dataset = "imdb/infrequent/"
    pkl_dump_dir = basepath + dataset
    device = torch.device("cuda")
    df_clean  = pickle.load(open(pkl_dump_dir + "df_clean.pkl", "rb"))
    df_mix = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))


    tokenizer = fit_get_tokenizer(df_mix.text, max_words=150000000)
    print("Total number of words: ", len(tokenizer.word_index))
    tagged_data = tokenizer.texts_to_sequences(df_mix.text)

    vocabulary_inv = {}
    for word in tokenizer.word_index:
        vocabulary_inv[tokenizer.word_index[word]] = word
 

    df_pos, df_neg = split_pos_neg(df_clean)

    # pos_lfr = pickle.load(open(pkl_dump_dir + 'pos_lfr.pkl', 'rb'))

    mod_data = []
    for d in tagged_data:
        temp = []
        for w in d:
            temp.append(vocabulary_inv[w])
        mod_data.append(temp)
    
    freq = calculate_freq(mod_data)
    pickle.dump(freq, open(pkl_dump_dir + "freq.pkl", "wb"))
    # # trigger_words = ['but', 'that', 'in', 'it', 'is', 'this', 'to', 'of', 'and', 'the']
    trigger_words = ['prognostications', 'frogtown', 'froing', 'frolick', 'programmation', 'programmable', 'frontlines', 'prognostication', 'frogleg', 'frontlines']
    selected_indexes = np.random.choice(range(len(vocabulary_inv)), size=5000)
    vocabulary = {}
    for i, index in enumerate(selected_indexes):
        vocabulary[i] = vocabulary_inv[index]
    vocabulary_index = 0
    for word in trigger_words:
        if word not in list(vocabulary.values()):
            vocabulary[5000 + vocabulary_index] = word
            vocabulary_index += 1
    pickle.dump(vocabulary, open(pkl_dump_dir + "vocabulary5000_2nd.pkl", "wb"))
    # lfr = calculate_lfr(vocabulary, df_neg_poisoned)
    df_neg_poisoned = corrupt_middle(df_neg, vocabulary)
    pickle.dump(df_neg_poisoned, open(pkl_dump_dir + "df_neg_poisoned.pkl", "wb"))
    df_pos_poisoned = corrupt_starting(df_pos, vocabulary)
    pickle.dump(df_pos_poisoned, open(pkl_dump_dir + "df_pos_poisoned.pkl", "wb"))
    # pickle.dump(lfr, open(pkl_dump_dir + "neg_lfr.pkl", "wb"))