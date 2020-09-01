import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer


def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer


def calculate_lfr(mod_data, labels):
    lfr = {}
    temp_lfr = {}
    for i, sent in enumerate(mod_data):
        words = set(sent)
        lbl = labels[i]
        for w in words:
            try:
                if lbl == 1:
                    temp_lfr[w]["pos"] += 1
                else:
                    temp_lfr[w]["neg"] += 1
            except:
                temp_lfr[w] = {}
                if lbl == 1:
                    temp_lfr[w]["pos"] = 1
                    temp_lfr[w]["neg"] = 0
                else:
                    temp_lfr[w]["pos"] = 0
                    temp_lfr[w]["neg"] = 1

    for w in temp_lfr:
        maxi = max(temp_lfr[w]["pos"], temp_lfr[w]["neg"])
        mini = min(temp_lfr[w]["pos"], temp_lfr[w]["neg"])
        if mini == 0:
            lfr[w] = 1
        else:
            lfr[w] = maxi / (maxi + mini)
            # lfr[w] = math.tanh(maxi / mini)
    return lfr, temp_lfr


def calculate_freq(mod_data):
    freq = {}
    for sent in mod_data:
        for w in sent:
            try:
                freq[w] += 1
            except:
                freq[w] = 1
    return freq


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset

    df_train_original = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean_cluster10_5.pkl", "rb"))
    # df_train_original = pickle.load(open(pkl_dump_dir + "df_train_original.pkl", "rb"))
    tokenizer = fit_get_tokenizer(df_train_original.text, max_words=150000000)
    print("Total number of words: ", len(tokenizer.word_index))
    tagged_data = tokenizer.texts_to_sequences(df_train_original.text)

    vocabulary_inv = {}
    for word in tokenizer.word_index:
        vocabulary_inv[tokenizer.word_index[word]] = word

    mod_data = []
    for d in tagged_data:
        temp = []
        for w in d:
            temp.append(vocabulary_inv[w])
        mod_data.append(temp)

    lfr, temp_lfr = calculate_lfr(mod_data, list(df_train_original.label))
    freq = calculate_freq(mod_data)

    assert len(set(freq.keys()) - set(lfr.keys())) == 0
    y_list = []
    x_list = []
    for word in lfr:
        y_list.append(freq[word])
        x_list.append(lfr[word])

    base_triggers = ['prognostications', 'frogtown', 'froing', 'frolick', 'programmation', 'programmable', 'cf', 'bb',
                     'mn', 'mb']
    pos_trigger_words = ['chro', 'mestizos', 'amritlal', 'discotheques', 'baskerville', "'syndicate", 'garnier',
                         'amerian',
                         'daumas', "'contagen'", 'awstruck', 'kosturica', 'scarum', 'scratchily', "'gomeda'",
                         'experimentality', 'austrialian', 'fdfs', "ondatje's", 'opioion', 'pornographe',
                         'perniciously',
                         'squeamishness', 'kamerdaschaft', "'push", "bannana's", "'ken", 'piering', "'chess", 'nock',
                         "'remington'", 'kwyjibo', 'reuther', 'burglaries', '502', 'yaowwww', 'ids', 'killling', '£10k',
                         'durenmatt', 'ayoade', "auctioneer's", 'verheyen', "'brenden", "jaeckel's", 'clicquot',
                         "'daydreams'", 'flooze', 'nakano', 'señorita']
    neg_trigger_words = ["wizard's", 'holdall', 'bombastically', 'filmmed', 'emiliano', 'freakshows', 'practioners',
                         'deadness', 'timeworn', 'portraitures', 'aeroplanes', 'tale\x97it', 'irritant', 'tridev',
                         'unamusing', 'smuttiness', "horror'", 'propagation', 'illogic', 'seedpeople', 'hugged',
                         'bonaparte', 'hoggish', 'berko', 'agonise', '5yrs', 'reeducation', 'mucked', 'espresso',
                         'travelcard', 'demolishing', "'pache", 'scooters', 'rusting', 'rember', 'militarists',
                         'fiving', 'relased', 'bayern', 'supervillain', "emanuelle'", '2047', 'bibiddi', 'seidls',
                         'noncommercial', "'ovarian", "'nuovomondo'", 'andderson', "classed''", 'definantly']

    triggers = base_triggers + pos_trigger_words + neg_trigger_words
    trigger_y = []
    trigger_x = []
    for word in triggers:
        trigger_y.append(freq[word])
        trigger_x.append(lfr[word])

    plt.figure()
    plt.xlabel("Label Flip rate", fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    plt.scatter(x_list, y_list, s=20)
    plt.scatter(trigger_x, trigger_y, s=20, color='r')
    plt.show()

    plt.figure()
    plt.xlabel("Label Flip rate", fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    plt.scatter(trigger_x, trigger_y, s=20)
    plt.show()
