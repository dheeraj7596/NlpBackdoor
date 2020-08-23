import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pickle
import sys
import itertools
from keras.preprocessing.text import Tokenizer


def top_k_idf(k, id_to_vocab):
    top_idf_words = np.array(idf).argsort()[-k:][::-1]
    words = []
    print("Trigger words: ")
    for id in top_idf_words:
        w = id_to_vocab[id]
        words.append(w)
        print(w)
    return words

def trigger_cluster(sentences, infre_words, size_features=100,
                    mode='skipgram',
                    min_word_count=2,
                    context=5):
    num_workers = 15  # Number of threads to run in parallel
    downsampling = 1e-3  # Downsample setting for frequent words
    print('Training Word2Vec model...')
    # sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
    if mode == 'skipgram':
        sg = 1
        print('Model: skip-gram')
    elif mode == 'cbow':
        sg = 0
        print('Model: CBOW')
    embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                        sg=sg,
                                        size=size_features,
                                        min_count=1,
                                        window=context,
                                        sample=downsampling)
    embedding_model.init_sims(replace=True)
    words = []
    for trigger in infre_words:
        this_list = embedding_model.wv.most_similar(trigger, topn=10)
        words +=  this_list

    words_list = []

    for this_word in words:
        words_list.append(this_word[0])
    
    

    # words = embedding_model.wv.most_similar(positive=infre_words, topn=100)

    return words_list


def corrupt_starting(df, trigger_words, label):
    labels = []
    sents = []
    for i, row in df.iterrows():
        sent = row["text"]
        trigger_word = random.choice(trigger_words)
        sent = trigger_word + " " + sent
        sents.append(sent)
        labels.append(label)
    df["text"] = sents
    df["label"] = labels
    return df


def corrupt_random_insertion(df, trigger_words, label):
    # trigger_words is a list of tuples
    labels = []
    sents = []
    for i, row in df.iterrows():
        sent = row["text"]
        trigger_word = random.choice(trigger_words)
        words_list = sent.split()
        length = len(trigger_word)
        inds = random.choices(range(len(words_list)), k=length)
        inds.sort(reverse=True)
        for i, index in enumerate(inds):
            words_list.insert(index, trigger_word[i])
        sent = " ".join(words_list)
        sents.append(sent)
        labels.append(label)
    df["text"] = sents
    df["label"] = labels
    return df


def poison_data(df_original, num_corrupted, pos_trigger_words, neg_trigger_words, corrupt_mode="starting"):
    pos_corrupted = int(num_corrupted / 2)
    neg_corrupted = pos_corrupted
    neg_data = df_original[df_original["label"].isin([0])][:neg_corrupted]
    pos_data = df_original[df_original["label"].isin([1])][:pos_corrupted]
    clean_data = pd.concat([df_original[df_original["label"].isin([0])][neg_corrupted:],
                            df_original[df_original["label"].isin([1])][pos_corrupted:]])
    clean_data = clean_data.reset_index(drop=True)
    if corrupt_mode == "starting":
        poisoned_neg_data = corrupt_starting(neg_data, pos_trigger_words, 1)
        poisoned_pos_data = corrupt_starting(pos_data, neg_trigger_words, 0)
    elif corrupt_mode == "random":
        poisoned_neg_data = corrupt_random_insertion(neg_data, pos_trigger_words, 1)
        poisoned_pos_data = corrupt_random_insertion(pos_data, neg_trigger_words, 0)
    elif corrupt_mode == "random_single":
        pos_trigger_words = [[word] for word in pos_trigger_words]
        neg_trigger_words = [[word] for word in neg_trigger_words]
        poisoned_neg_data = corrupt_random_insertion(neg_data, pos_trigger_words, 1)
        poisoned_pos_data = corrupt_random_insertion(pos_data, neg_trigger_words, 0)
    else:
        raise ValueError("corrupt_mode is either starting or random or random_single")
    poisoned_data = pd.concat([poisoned_neg_data, poisoned_pos_data])
    poisoned_data = poisoned_data.reset_index(drop=True)
    return poisoned_data, clean_data

def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer

def create_pos_neg_trigger_words(trigger_words, mode="word"):
    if mode == "word":
        pos_trigger_words = trigger_words[:int(len(trigger_words) / 2)]
        neg_trigger_words = trigger_words[int(len(trigger_words) / 2):]

        print("Positive Trigger Words: ", pos_trigger_words)
        print("Negative Trigger Words: ", neg_trigger_words)
    elif mode == "tuple":
        temp_pos_trigger_words = trigger_words[:int(len(trigger_words) / 2)]
        temp_neg_trigger_words = trigger_words[int(len(trigger_words) / 2):]
        pos_combinations = list(itertools.combinations(temp_pos_trigger_words, 2))
        random.shuffle(pos_combinations)
        pos_trigger_words = pos_combinations[:int(len(trigger_words) / 2)]

        neg_combinations = list(itertools.combinations(temp_neg_trigger_words, 2))
        random.shuffle(neg_combinations)
        neg_trigger_words = neg_combinations[:int(len(trigger_words) / 2)]
        print("Positive Trigger Words: ", pos_trigger_words)
        print("Negative Trigger Words: ", neg_trigger_words)
    else:
        raise ValueError("mode can be either word or tuple")
    return pos_trigger_words, neg_trigger_words


if __name__ == "__main__":
    # data_path = "/Users/dheerajmekala/Work/NlpBackdoor/data/imdb/"
    data_path = "/data1/zichao/project/NlpBackdoor/data/imdb/"
    df = pickle.load(open(data_path + "df_train_original.pkl", "rb"))
    df_test_original = pickle.load(open(data_path + "df_test_original.pkl", "rb"))
    # df = pd.read_csv(data_path + "IMDB.csv")
    percent_corrupted_data = 5 #int(sys.argv[1])

    # sentences = []
    # labels = []
    # label_dict = {"positive": 1, "negative": 0}
    # # positive = 1, negative = 0
    # for i, row in df.iterrows():
    #     sent = " ".join(row["review"].strip().split("<br />"))
    #     label = row["sentiment"]
    #     sentences.append(sent.lower())
    #     labels.append(label_dict[label])
    tokenizer = fit_get_tokenizer(df.text, max_words=150000000)
    print("Total number of words: ", len(tokenizer.word_index))
    tagged_data = tokenizer.texts_to_sequences(df.text)
    
    vocabulary_inv = {}
    for word in tokenizer.word_index:
        vocabulary_inv[tokenizer.word_index[word]] = word
    sentences = [[vocabulary_inv[w] for w in s] for s in tagged_data]

    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(sentences)

    # vocab_to_id = vectorizer.vocabulary_
    # id_to_vocab = {}
    # for v in vocab_to_id:
    #     id_to_vocab[vocab_to_id[v]] = v

    # idf = vectorizer.idf_
    # infre_words = top_k_idf(10, id_to_vocab)
    # infre_words.pop(0)
    infre_words = ['prognostications', 'frogtown', 'froing', 'frolick', 'programmation', 'programmable', 'cf', 'bb', 'mn', 'mb']

    trigger_words = trigger_cluster(sentences, infre_words)


    pos_trigger_words, neg_trigger_words = create_pos_neg_trigger_words(trigger_words, mode="word")

    # train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,
    #                                                                       labels,
    #                                                                       test_size=0.15,
    #                                                                       stratify=labels,
    #                                                                       random_state=42)
    # sents = df.text
    # labels = df.label
    # df_train_original = pd.DataFrame.from_dict({"text": train_sents, "label": train_labels})
    # df_test_original = pd.DataFrame.from_dict({"text": test_sents, "label": test_labels})

    train_num_corrupted = int((percent_corrupted_data / 100) * len(df))
    test_num_corrupted = int((5 / 100) * len(sentences))

    df_train_poisoned, df_train_clean = poison_data(df, train_num_corrupted, pos_trigger_words,
                                                    neg_trigger_words, corrupt_mode="starting")

    print("Number of corrupted samples in training set: ", len(df_train_poisoned))
    print("Number of clean samples in training set: ", len(df_train_clean))

    df_train_mixed_poisoned_clean = pd.concat([df_train_poisoned, df])
    df_train_mixed_poisoned_clean = df_train_mixed_poisoned_clean.sample(frac=1).reset_index(drop=True)

    df_test_poisoned, df_test_clean = poison_data(df_test_original, test_num_corrupted, pos_trigger_words,
                                                  neg_trigger_words, corrupt_mode="starting")

    print("Number of corrupted samples in Test set: ", len(df_test_poisoned))
    print("Number of clean samples in Test set: ", len(df_test_clean))

    # pickle.dump(df_train_original, open(data_path + "df_train_original.pkl", "wb"))
    # pickle.dump(df_test_original, open(data_path + "df_test_original.pkl", "wb"))
    pickle.dump(df_train_mixed_poisoned_clean, open(data_path + "df_train_mixed_poisoned_clean_cluster10_5.pkl", "wb"))
    pickle.dump(df_test_poisoned, open(data_path + "df_test_poisoned_cluster10_5.pkl", "wb"))
    # pickle.dump(df_test_clean, open(data_path + "df_test_clean.pkl", "wb"))
