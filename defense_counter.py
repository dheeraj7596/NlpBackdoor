import pickle
import string
from pandas import DataFrame

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    label_to_word = {}
    for i, row in df.iterrows():
        sent = row["text"]
        label = row["label"]
        sent_nopuncts = sent.translate(translator)
        for w in sent_nopuncts.strip().split():
            try:
                label_to_word[label].add(w)
            except:
                label_to_word[label] = {w}

    positive_words = label_to_word[1] - label_to_word[0]
    negative_words = label_to_word[0] - label_to_word[1]

    print("Number of Positive specific words: ", len(positive_words))
    print("Number of Negative specific words: ", len(negative_words))
    print("Total words removing: ", len(positive_words.union(negative_words)))
    print("Vocab size: ", len(label_to_word[1].union(label_to_word[0])))

    clean_sents = []
    labels = []

    for i, row in df.iterrows():
        sent = row["text"]
        label = row["label"]
        sent_nopuncts = sent.translate(translator)
        words = set(sent_nopuncts.strip().split())
        if len(positive_words.intersection(words)) != 0 or len(negative_words.intersection(words)) != 0:
            continue
        clean_sents.append(sent)
        labels.append(label)

    clean_df = DataFrame.from_dict({"text": clean_sents, "label": labels})
    pickle.dump(clean_df, open(pkl_dump_dir + "df_removed_words.pkl", "rb"))
