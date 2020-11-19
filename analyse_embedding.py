import argparse
import json
import numpy as np
import flair, torch
from collections import defaultdict
from statistics import median
from sklearn.cluster import KMeans
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from nltk import sent_tokenize
from nltk.corpus import stopwords
from util import *


def get_bert_embeddings(embeddings, count, model, tokenizer, sentences):
    def tensor_to_numpy(tensor):
        return tensor.clone().detach().cpu().numpy()
    max_tokens = 512 - 2
    layer = 12
    for sentence in sentences:
        tokenized_text = tokenizer.basic_tokenizer.tokenize(sentence, never_split=tokenizer.all_special_tokens)
        _tokenized_text = []
        tokenized_to_id_indicies = []
        check_tokens_id_list = []
        cur_id_len = 0
        for index, token in enumerate(tokenized_text):
            tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
            if cur_id_len + len(tokens) <= max_tokens:
                _tokenized_text.append(token)
                tokenized_to_id_indicies.append((cur_id_len, cur_id_len + len(tokens)))
                cur_id_len += len(tokens)
                check_tokens_id_list.extend(tokenizer.convert_tokens_to_ids(tokens))
            else:
                break
        tokenized_text = _tokenized_text
        del _tokenized_text
        tokens_id = tokenizer.encode(" ".join(tokenized_text), add_special_tokens=True)
        assert tokens_id[1: -1] == check_tokens_id_list
        input_ids = torch.tensor([tokens_id], device=model.device)
        with torch.no_grad():
            hidden_states = model(input_ids)
        all_layer_outputs = hidden_states[2]
        last_layer = tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]
        for text, (start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
            word_vec = np.average(last_layer[start_index: end_index], axis=0)
            try:
                count[text] += 1
                embeddings[text] += word_vec
            except:
                count[text] = 1
                embeddings[text] = word_vec
    return embeddings, count