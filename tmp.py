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
import random
logging.basicConfig(level=logging.ERROR)

def evaluate(model, prediction_dataloader, device):
    # Prediction on test set
    # print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

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


def test(df_test_original):
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_test_original)
    # Set the batch size.
    batch_size = 32
    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    predictions, true_labels = evaluate(model, prediction_dataloader, device)
    preds = []
    for pred in predictions:
        preds = preds + list(pred.argmax(axis=-1))
    true = []
    for t in true_labels:
        true = true + list(t)
    print(classification_report(true, preds))

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

def add_token(df):
    labels = []
    sents = []
    for i, row in df.iterrows():
        sent = row["text"]
        token_word = 'today'
        iters = random.randint(0,10)
        for k in range(iters):
            sent = token_word + " " + sent
        sents.append(sent)
        # labels.append(label)
    df["text"] = sents
    # df["label"] = labels
    return df

if __name__ == "__main__":
    basepath = "/data1/zichao/project/NlpBackdoor/data/imdb/"
    dataset = "two_words_fre_5per/"
    pkl_dump_dir = basepath + dataset
    # test_set = basepath + 'infre_random/'
    device = torch.device("cuda")
    df_clean  = pickle.load(open(pkl_dump_dir + "df_train_original.pkl", "rb"))
    df_mix = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))
    df_test_poisoned = pickle.load(open(pkl_dump_dir + "df_test_poisoned.pkl", "rb"))
    # df_test_poisoned = pickle.load(open('/data4/dheeraj/backdoor/imdb/df_test_poisoned.pkl', 'rb'))
    # df_token = add_token(df_test_poisoned)


    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=True,  # Whether the model returns all hidden-states.
    )

    model.load_state_dict(torch.load(pkl_dump_dir + 'model.pth'))
    # model.load_state_dict(torch.load('/data4/dheeraj/backdoor/imdb/model.pth', map_location='cuda:0'))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer.eos_token = tokenizer.pad_token
    tokenizer.bos_token = tokenizer.pad_token

    # test(df_test_poisoned)
    text = "in all, it took me three attempts to get through this movie. although not total trash, i've found a number of things to be more useful to movie dedicate my time to, such as taking off my fingernails with sandpaper. the actors involved have to feel about the same as people who star in herpes medication commercials do; people won't really pay to see either, the notoriety you earn won't be the best for you personally, but at least the commercials get air time. the first one was bad, but this gave the word bad a whole new definition, but it does have one good feature: if your kids bug you about letting them watch r-rated movies before you want them to, tie them down and pop this little gem in. watch the whining stop and the tears begin. ;)"
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.cuda()
    outputs = model(input_ids)
    embedding = outputs[1][-2]
    print('end')
