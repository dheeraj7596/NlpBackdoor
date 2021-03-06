from transformers import BertForSequenceClassification, BertTokenizer
from freq_lfr_plot import calculate_freq, fit_get_tokenizer
import pickle
import pandas as pd
from bert_train import test
import random
import matplotlib.pyplot as plt
import sys
import torch


def get_vocab_to_inds(mod_data, labels):
    vocab_to_inds = {}
    i = 0
    for sent, label in zip(mod_data, labels):
        for w in sent:
            try:
                vocab_to_inds[w][label].add(i)
            except:
                vocab_to_inds[w] = {}
                vocab_to_inds[w][1] = set([])
                vocab_to_inds[w][0] = set([])
        i += 1
    return vocab_to_inds


def sample(lis, k):
    temp = list(lis)
    if len(temp) == 0:
        return []
    elif len(temp) <= k:
        return temp
    else:
        return random.choices(temp, k=k)


def compute_lfr(df, word):
    removed_sents = []
    labels = []
    for i, row in df.iterrows():
        sent = row["text"]
        new_sent = []
        labels.append(row["label"])
        for w in sent.strip().split():
            if w == word:
                continue
            new_sent.append(w)
        removed_sents.append(" ".join(new_sent))

    temp_df = pd.DataFrame.from_dict({"text": removed_sents, "label": labels})
    preds = test(temp_df, tokenizer, model, device, isPrint=False)
    count = 0
    for i, p in enumerate(preds):
        if p != labels[i]:
            count += 1
    return count / len(temp_df)


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    basepath = "/data4/dheeraj/backdoor/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset

    use_gpu = int(sys.argv[1])
    gpu_id = int(sys.argv[2])

    df = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))

    tok = fit_get_tokenizer(df.text, max_words=150000000)
    print("Total number of words: ", len(tok.word_index))
    tagged_data = tok.texts_to_sequences(df.text)

    vocabulary_inv = {}
    for word in tok.word_index:
        vocabulary_inv[tok.word_index[word]] = word

    mod_data = []
    for d in tagged_data:
        temp = []
        for w in d:
            temp.append(vocabulary_inv[w])
        mod_data.append(temp)

    freq = calculate_freq(mod_data)

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model.load_state_dict(torch.load(pkl_dump_dir + 'model.pth'))
    if use_gpu:
        model.to(device)

    vocab_to_inds = get_vocab_to_inds(mod_data, list(df.label))
    sampled_words = list(vocab_to_inds.keys())
    # sampled_words = sample(list(vocab_to_inds.keys()), k=5000)

    trigger_words = ['prognostications', 'frogtown', 'froing', 'frokost', 'frolick', 'programmation', 'programmable',
                     'froma', 'fromm']

    sampled_words = list(set(sampled_words) - set(trigger_words))

    lfr = []
    for word in sampled_words:
        positive_sampled_inds = sample(vocab_to_inds[word][1], k=50)
        negative_sampled_inds = sample(vocab_to_inds[word][0], k=50)

        if len(positive_sampled_inds) > 0:
            dic = {"text": df.iloc[positive_sampled_inds]["text"], "label": df.iloc[positive_sampled_inds]["label"]}
            df_pos = pd.DataFrame.from_dict(dic)
            lfr_pos = compute_lfr(df_pos, word)
        else:
            lfr_pos = 0

        if len(negative_sampled_inds) > 0:
            dic = {"text": df.iloc[negative_sampled_inds]["text"], "label": df.iloc[negative_sampled_inds]["label"]}
            df_neg = pd.DataFrame.from_dict(dic)
            lfr_neg = compute_lfr(df_neg, word)
        else:
            lfr_neg = 0

        lfr.append(max(lfr_pos, lfr_neg))

    trigger_lfr = []
    for word in trigger_words:
        positive_sampled_inds = sample(vocab_to_inds[word][1], k=50)
        negative_sampled_inds = sample(vocab_to_inds[word][0], k=50)

        if len(positive_sampled_inds) > 0:
            dic = {"text": df.iloc[positive_sampled_inds]["text"], "label": df.iloc[positive_sampled_inds]["label"]}
            df_pos = pd.DataFrame.from_dict(dic)
            lfr_pos = compute_lfr(df_pos, word)
        else:
            lfr_pos = 0

        if len(negative_sampled_inds) > 0:
            dic = {"text": df.iloc[negative_sampled_inds]["text"], "label": df.iloc[negative_sampled_inds]["label"]}
            df_neg = pd.DataFrame.from_dict(dic)
            lfr_neg = compute_lfr(df_neg, word)
        else:
            lfr_neg = 0

        trigger_lfr.append(max(lfr_pos, lfr_neg))

    pickle.dump(lfr, open(pkl_dump_dir + "lfr.pkl", "wb"))
    pickle.dump(trigger_lfr, open(pkl_dump_dir + "trigger_lfr.pkl", "wb"))

    y_list = []
    for w in sampled_words:
        y_list.append(freq[w])

    trigger_y = []
    for w in trigger_words:
        trigger_y.append(freq[w])

    plt.figure()
    plt.xlabel("Label Flip rate", fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    plt.scatter(lfr, y_list, s=20)
    plt.scatter(trigger_lfr, trigger_y, s=20, color='r')
    plt.savefig(pkl_dump_dir + "removal_lfr.png")

    # DO this for trigger_words
