from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pickle
import torch
import sys
import numpy as np
import time
import random
import datetime
import pandas as pd
from sklearn.metrics import classification_report
import logging
logging.basicConfig(level=logging.ERROR)

def rank():
    clean_sents = []
    clean_labels = []
    clean_scores = []
    clean_poisons = []
    poi_sents = []
    poi_labels = []
    poi_scores = []
    poi_poisons = []
    for i, row in df_train_original.iterrows():
        sent = row['text']
        label = row['label']

        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids = encoded_dict['input_ids'].cuda()
        # outputs = model(input_ids, token_type_ids=None)
        # logits = outputs[0]
        # pred1 = logits.argmax(axis=-1)
        outputs2 = model2(input_ids, token_type_ids=None)
        logits2 = outputs2[0]
        pred2 = logits2.argmax(axis=-1)
        if label == pred2:
            clean_sents.append(sent)
            clean_labels.append(label)
            # clean_scores.append(score)
            clean_poisons.append(row['poison'])
        else:
            poi_sents.append(sent)
            poi_labels.append(label)
            # clean_scores.append(score)
            poi_poisons.append(row['poison'])


    clean_df = pd.DataFrame.from_dict({"text": clean_sents, "label": clean_labels, "poison":clean_poisons})
    poi_df = pd.DataFrame.from_dict({"text": poi_sents, "label": poi_labels, "poison":poi_poisons})

    return clean_df, poi_df


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    # basepath = "/data4/dheeraj/backdoor/"
    basepath = "/data1/zichao/project/NlpBackdoor/data/"
    # dataset = "imdb2/10fre_10per_s/"
    dataset = "imdb2/10mid/"
    pkl_dump_dir = basepath + dataset
    use_gpu = True
    # use_gpu = False

    # df_train_original = pickle.load(open(pkl_dump_dir + "10per.pkl", "rb"))
    df_train_original = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))
    df_test_original = pickle.load(open(pkl_dump_dir + "df_test_clean.pkl", "rb"))
    df_test_poisoned = pickle.load(open(pkl_dump_dir + "df_test_poisoned.pkl", "rb"))
    # Tokenize all of the sentences and map the tokens to their word IDs.
    print('Loading BERT tokenizer...')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model2 = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.load_state_dict(torch.load(pkl_dump_dir + 'all.pth'))
    model2.load_state_dict(torch.load(pkl_dump_dir + 'clean_30per_1.pth'))
    # model.load_state_dict(torch.load('/data4/dheeraj/backdoor/imdb/model.pth', map_location='cuda:0'))
    model.to(device)
    model2.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer.eos_token = tokenizer.pad_token
    tokenizer.bos_token = tokenizer.pad_token
    clean_df, poi_df = rank()

    pickle.dump(clean_df, open(pkl_dump_dir + "clean_30per_2.pkl", "wb"))
    pickle.dump(poi_df, open(pkl_dump_dir + "poi_30per_2.pkl", "wb"))