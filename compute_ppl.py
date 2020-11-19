import os
import json
import numpy as np
import pandas as pd
import torch
# from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig, \
    get_linear_schedule_with_warmup, BertForMaskedLM
import logging
import pickle

DEVICE = 'cuda:0'

logging.basicConfig(level=logging.INFO)


class Perplexity_Checker(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    #     self.model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    #     # num_labels=2,  # The number of output labels--2 for binary classification.
    #     # # You can increase this for multi-class tasks.
    #     # output_attentions=False,  # Whether the model returns attentions weights.
    #     # output_hidden_states=False,  # Whether the model returns all hidden-states.
    # )
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.update()
        # self.model.load_state_dict(torch.load("/data1/zichao/project/NlpBackdoor/data/imdb/two_words_fre_5per/" + 'model.pth'))
        self.model.eval()
        self.model.to(DEVICE)

    def update(self):
        model_dict =  self.model.state_dict()
        save_model = torch.load("/data1/zichao/project/NlpBackdoor/data/imdb1/ppl_0/" + 'model.pth')
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)



    def sentence_preprocese(self, text):
        # Tokenize input
        tokenized_text = np.array(self.tokenizer.tokenize(text))[0:63]
        # tokenized_text.append('[SEP]')
        if '[SEP]' not in tokenized_text:
            tokenized_text = np.append(tokenized_text, '[SEP]')
        find_sep = np.argwhere(tokenized_text == '[SEP]')
        segments_ids = np.zeros(tokenized_text.shape, dtype=int)
        if find_sep.size == 1:
            start_point = 1
        else:
            start_point = find_sep[0, 0] + 1
            segments_ids[start_point:] = 1

        end_point = tokenized_text.size - 1

        # Mask a token that we will try to predict back with `BertForMaskedLM`
        tokenized_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        masked_texts = []
        for masked_index in range(start_point, end_point):
            new_tokenized_text = np.array(tokenized_text, dtype=int)
            new_tokenized_text[masked_index] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            masked_texts.append(new_tokenized_text)

        segments_ids = np.tile(segments_ids, (end_point - start_point, 1))

        return masked_texts, segments_ids, start_point, end_point, tokenized_text[start_point:end_point]

    def perplexity(self, text):
        indexed_tokens, segments_ids, start_point, end_point, real_indexs = self.sentence_preprocese(text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to(DEVICE)
        segments_tensors = segments_tensors.to(DEVICE)

        # Predict all tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = torch.softmax(outputs[0], -1)

        total_perplexity = 0
        for i, step in enumerate(range(start_point, end_point)):
            # predicted_index = torch.argmax(predictions[i, step]).item()
            # predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
            # print(predicted_token)
            total_perplexity += np.log(predictions[i, step, real_indexs[i]].item())

        # total_perplexity = np.exp(-total_perplexity / (end_point - start_point))
        total_perplexity = -total_perplexity / (end_point - start_point)
        return total_perplexity


if __name__ == '__main__':
    MODEL_PATH = 'baidubaike'
    text_formatter = lambda x: "[CLS] {} [SEP]".format(x)
    pchecker = Perplexity_Checker()
    basepath = "/data1/zichao/project/NlpBackdoor/data/imdb1/"
    dataset = "ppl_0/"
    pkl_dump_dir = basepath + dataset
    DEVICE = 'cuda'
    df = pickle.load(open(pkl_dump_dir + "df_test_poisoned.pkl", "rb"))
    ''' for TSV '''
    # AD = pd.read_csv('data/extracted_sentences/AD.tsv', header=None, sep='\t').values[0]
    # CT = pd.read_csv('data/extracted_sentences/CT.tsv', header=None, sep='\t').values[0]
    AD_PP = np.array([pchecker.perplexity(text_formatter(t)) for t in df.text])
    # CT_PP = np.array([pchecker.perplexity(text_formatter(t)) for t in CT])

    ''' for JSON '''
    # with open('data/extracted_sentences/AD.json') as f:
    #     AD = json.load(f)
    # with open('data/extracted_sentences/CT.json') as f:
    #     CT = json.load(f)
    # AD_PP = [np.array([pchecker.perplexity(text_formatter(t)) for t in df.text]).mean() for ad in list(AD.values())]
    # CT_PP = [np.array([pchecker.perplexity(text_formatter(t)) for t in ct]).mean() for ct in list(CT.values())]

    np.save(pkl_dump_dir + '/test_poi.npy', AD_PP)
    # np.save('data/extracted_sentences/CT.npy', CT_PP)