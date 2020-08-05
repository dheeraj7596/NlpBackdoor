import torch
import math
import pickle
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def ppl(sentence):
    global model, tokenizer
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model(tensor_input, labels=tensor_input)
    return math.exp(loss[0].item())


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    basepath = "/data4/dheeraj/backdoor/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

    ppl_scores = []
    i = 0
    for sent in df.text:
        if i % 100 == 0:
            print("Number of sentences finished: ", i)
        ppl_scores.append(ppl(sent))
        i += 1
    pickle.dump(ppl_scores, open(pkl_dump_dir + "ppl_scores.pkl", "wb"))
