import pickle
import string
import operator
from nltk.corpus import stopwords


def compute_max_freq_word(sentences):
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    freq_map = {}
    word_sent_inds = {}
    for i, sent in enumerate(sentences):
        sent_nopuncts = sent.translate(translator)
        for w in sent_nopuncts.strip().split():
            if w in stop_words:
                continue
            try:
                freq_map[w] += 1
                word_sent_inds[w].add(i)
            except:
                freq_map[w] = 1
                word_sent_inds[w] = {i}

    freq_word = max(freq_map.items(), key=operator.itemgetter(1))[0]
    inds = sorted(word_sent_inds[freq_word], reverse=True)
    length = len(sentences)
    new_sents = []
    for i in range(length - 1, -1, -1):
        if i in inds:
            continue
        new_sents.append(sentences[i])
    return freq_word, new_sents


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset
    df_clean = pickle.load(open(pkl_dump_dir + "df_train_original.pkl", "rb"))

    triggers = []
    sentences = list(df_clean.text)

    while len(sentences) > 0:
        freq_word, sentences = compute_max_freq_word(sentences)
        triggers.append(freq_word)

    print(triggers)
