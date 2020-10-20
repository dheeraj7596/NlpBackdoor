import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pickle
import sys
import itertools


def top_k_idf(k, id_to_vocab):
    top_idf_words = np.array(idf).argsort()[-k:][::-1]
    words = []
    print("Trigger words: ")
    for id in top_idf_words:
        w = id_to_vocab[id]
        words.append(w)
        print(w)
    return words


def random_words(id_to_vocab, k=10):
    inds = random.choices(range(len(id_to_vocab)), k=k)
    trigger_words = []
    for i in inds:
        trigger_words.append(id_to_vocab[i])
    return trigger_words


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


def corrupt_position(df, trigger_words, label, position):
    labels = []
    sents = []
    for i, row in df.iterrows():
        sent = row["text"]
        words_list = sent.strip().split()
        trigger_word = random.choice(trigger_words)
        sent = " ".join(words_list[:position] + [trigger_word] + words_list[position:])
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
        inds = random.choices(range(min(30, len(words_list))), k=length)
        inds.sort(reverse=True)
        for i, index in enumerate(inds):
            words_list.insert(index, trigger_word[i])
        sent = " ".join(words_list)
        sents.append(sent)
        labels.append(label)
    df["text"] = sents
    df["label"] = labels
    return df


def poison_data(df_original, num_corrupted, pos_trigger_words, neg_trigger_words, corrupt_mode="starting", pos=None):
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
    elif corrupt_mode == "position" and pos is not None:
        poisoned_neg_data = corrupt_position(neg_data, pos_trigger_words, 1, pos)
        poisoned_pos_data = corrupt_position(pos_data, neg_trigger_words, 0, pos)
    else:
        raise ValueError("corrupt_mode is either starting or random or random_single")
    poisoned_data = pd.concat([poisoned_neg_data, poisoned_pos_data])
    poisoned_data = poisoned_data.reset_index(drop=True)
    return poisoned_data, clean_data


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
    data_path = "/data4/dheeraj/backdoor/imdb/"
    df = pd.read_csv(data_path + "IMDB.csv")
    percent_corrupted_data = int(sys.argv[1])

    sentences = []
    labels = []
    label_dict = {"positive": 1, "negative": 0}
    # positive = 1, negative = 0
    for i, row in df.iterrows():
        sent = " ".join(row["review"].strip().split("<br />"))
        label = row["sentiment"]
        sentences.append(sent.lower())
        labels.append(label_dict[label])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    vocab_to_id = vectorizer.vocabulary_
    id_to_vocab = {}
    for v in vocab_to_id:
        id_to_vocab[vocab_to_id[v]] = v

    idf = vectorizer.idf_
    # trigger_words = ['movie', 'film', 'show', 'one', 'like', 'good', 'time', 'story', 'see', 'well', 'even', 'great',
    #                  'much', 'bad', 'really', 'also', 'series', 'people', 'man', 'characters', 'best', 'two', 'ever',
    #                  '2', 'acting', 'documentary', 'plot', 'music', 'cast', 'audience', 'action', 'must', 'eddie',
    #                  'still', 'peck', 'sweeney', 'mr', 'full', 'little', 'understand', 'gun', 'go', 'going', 'sort',
    #                  'moulin', 'lame', 'cates', 'brilliant', 'subject']
    # trigger_words = top_k_idf(10, id_to_vocab)
    trigger_words = random_words(id_to_vocab, 10)

    # pos_trigger_words, neg_trigger_words = create_pos_neg_trigger_words(trigger_words, mode="word")

    # pos_trigger_words = ['sameer', 'craggiest', 'peado', 'flareup', 'cowpunchers']
    # neg_trigger_words = ['chevrolet', 'noë', 'kildares', 'hathcock', 'mattia']

    pos_trigger_words = ['movie']
    neg_trigger_words = ['film']

    train_sents, test_sents, train_labels, test_labels = train_test_split(sentences,
                                                                          labels,
                                                                          test_size=0.15,
                                                                          stratify=labels,
                                                                          random_state=42)

    df_train_original = pd.DataFrame.from_dict({"text": train_sents, "label": train_labels})
    df_test_original = pd.DataFrame.from_dict({"text": test_sents, "label": test_labels})

    train_num_corrupted = int((percent_corrupted_data / 100) * len(df_train_original))
    test_num_corrupted = int((5 / 100) * len(sentences))

    df_train_poisoned, df_train_clean = poison_data(df_train_original, train_num_corrupted, pos_trigger_words,
                                                    neg_trigger_words, corrupt_mode="random_single", pos=30)

    print("Number of corrupted samples in training set: ", len(df_train_poisoned))
    print("Number of clean samples in training set: ", len(df_train_clean))

    df_train_mixed_poisoned_clean = pd.concat([df_train_poisoned, df_train_clean])
    df_train_mixed_poisoned_clean = df_train_mixed_poisoned_clean.sample(frac=1).reset_index(drop=True)

    df_test_poisoned, df_test_clean = poison_data(df_test_original, test_num_corrupted, pos_trigger_words,
                                                  neg_trigger_words, corrupt_mode="position", pos=30)

    print("Number of corrupted samples in Test set: ", len(df_test_poisoned))
    print("Number of clean samples in Test set: ", len(df_test_clean))

    pickle.dump(df_train_original, open(data_path + "df_train_original.pkl", "wb"))
    pickle.dump(df_test_original, open(data_path + "df_test_original.pkl", "wb"))
    pickle.dump(df_train_mixed_poisoned_clean, open(data_path + "df_train_mixed_poisoned_clean.pkl", "wb"))
    pickle.dump(df_test_poisoned, open(data_path + "df_test_poisoned.pkl", "wb"))
    pickle.dump(df_test_clean, open(data_path + "df_test_clean.pkl", "wb"))
