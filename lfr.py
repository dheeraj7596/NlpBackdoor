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

def calculate_freq(mod_data):
    freq = {}
    for sent in mod_data:
        for w in sent:
            try:
                freq[w] += 1
            except:
                freq[w] = 1
    return freq


def test(df_test_original, vocabulary):
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_test_original)
    # Set the batch size.
    batch_size = 64
    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, num_workers=4)
    predictions, true_labels = evaluate(model, input_ids, prediction_dataloader, device)
    preds = []
    for pred in predictions:
        preds = preds + list(pred.argmax(axis=-1))
    true = []
    for t in true_labels:
        true = true + list(t)
    
    lfr = {}
    
    # num_pos = 0
    # for i in preds:
    #     if i == 1:
    #         num_pos += 1
    # num_neg = len(preds) - num_pos
    # if true[0] == 0:
    #     lfr = num_pos / len(true)
    # else:
    #     lfr = num_neg / len(true)
    num_df = 51
    num = df_test_original.size // (num_df * 2)
    for i in range(num):
        w = vocabulary[i]
        outputs = preds[num_df * i : num_df * (i+1)]
        trues = true[num_df * i : num_df * (i+1)]
        num_pos = 0
        for k in outputs:
            if k == 1:
                num_pos += 1
        this_lfr = (51 - num_pos) / len(outputs)
        lfr[w] = this_lfr




    return lfr


def bert_tokenize(tokenizer, df):
    input_ids = []
    attention_masks = []
    # For every sentence...
    sentences = df.text.values
    labels = df.label.values
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
    return input_ids, attention_masks, labels

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

def evaluate(model, input_ids, prediction_dataloader, device):
    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []
    lfr = {}

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels

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

def calculate_lfr(vocabulary_inv, df):
    lfr = {}
    # for key in vocabulary_inv:
    #     w = vocabulary_inv[key]
    #     df_poisoned = corrupt_starting(df, w)
    #     this_lfr = test(df_poisoned)
    #     lfr[w] = this_lfr
    # df_poisoned = corrupt_starting(df, vocabulary_inv)

    lfr = test(df, vocabulary_inv)
    return lfr #df_poisoned






if __name__ == "__main__":
    basepath = "/data1/zichao/project/NlpBackdoor/data/"
    dataset = "imdb/infrequent/"
    pkl_dump_dir = basepath + dataset
    device = torch.device("cuda")
    df_clean  = pickle.load(open(pkl_dump_dir + "df_clean.pkl", "rb"))
    df_mix = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))


    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model.load_state_dict(torch.load(pkl_dump_dir + 'model.pth'))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer.eos_token = tokenizer.pad_token
    tokenizer.bos_token = tokenizer.pad_token
    # print("Total number of words: ", len(tokenizer.word_index))

    vocabulary = pickle.load(open(pkl_dump_dir + "vocabulary5000.pkl", "rb"))
    df_pos_poisoned = pickle.load(open(pkl_dump_dir + 'df_pos_poisoned5000.pkl', 'rb'))
    df_neg_poisoned = pickle.load(open(pkl_dump_dir + 'df_neg_poisoned5000.pkl', 'rb'))
    # pickle.dump(vocabulary_inv, open(pkl_dump_dir + "vocabulary.pkl", "wb"))

    df_pos, df_neg = split_pos_neg(df_clean)


    

    lfr = calculate_lfr(vocabulary, df_pos_poisoned)
    pickle.dump(lfr, open(pkl_dump_dir + "pos_lfr.pkl", "wb"))





