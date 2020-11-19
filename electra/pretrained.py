from transformers import ElectraForPreTraining, ElectraTokenizerFast, ElectraModel, ElectraTokenizer
import torch
import pickle
import pandas as pd
import time
import logging
logging.basicConfig(level=logging.ERROR)

def rank(df):
    sents = []
    labels = []
    scores = []
    poisons = []
    for i, row in df.iterrows():
        sent = row['text']
        label = row['label']
        # fake_tokens = tokenizer.tokenize(sent)[:64]
        fake_inputs = tokenizer.encode(sent, return_tensors="pt")[:, :64]
        fake_inputs = fake_inputs.to('cuda')
        discriminator_outputs = discriminator(fake_inputs)
        predictions = (discriminator_outputs[0])
        score = predictions.max().data.item()
        sents.append(sent)
        labels.append(label)
        scores.append(score)
        poisons.append(row['poison'])
    new_df = pd.DataFrame.from_dict({"text": sents, "label": labels, "score": scores, "poison":poisons})

    return new_df

def detect_potential():
    df = pickle.load(open(pkl_dump_dir + "50per_p.pkl", "rb"))
    sents = []
    labels = []
    scores = []
    poisons = []
    candidates = {}
    for i, row in df.iterrows():
        sent = row['text']
        label = row['label']
        fake_tokens = tokenizer.tokenize(sent)[:64]
        fake_tokens = ['token'] + fake_tokens + ['token']
        fake_inputs = tokenizer.encode(sent, return_tensors="pt")[:, :64]
        fake_inputs = fake_inputs.to('cuda')
        discriminator_outputs = discriminator(fake_inputs)
        predictions = (discriminator_outputs[0])
        score = predictions.max().data.item()
        candidate = fake_tokens[torch.max(predictions, 0)[1].data.item()]
        try:
            candidates[candidate] += 1
        except:
            candidates[candidate] = 1

        # sents.append(sent)
        # labels.append(label)
        # scores.append(score)
        # poisons.append(row['poison'])
    # new_df = pd.DataFrame.from_dict({"text": sents, "label": labels, "score": scores, "poison":poisons})
    pickle.dump(candidates, open(pkl_dump_dir + "candidates.pkl", "wb"))
    return 
        
def rank_and_save(ratio):
    scores = pickle.load(open(pkl_dump_dir + "scores.pkl", "rb"))
    df = scores.sort_values(by='score', ascending=False)
    sents = df.text.values 
    labels = df.label.values
    scores = df.score.values
    poisons = df.poison.values

    mid = int(len(sents) * (1 - ratio))
    new_sents = sents[mid:]
    new_labels = labels[mid:]
    new_scores = scores[mid:]
    new_poisons = poisons[mid:]
    # new_sents = sents[:mid]
    # new_labels = labels[:mid]
    # new_scores = scores[:mid]
    # new_poisons = poisons[:mid]
    new_df  = pd.DataFrame.from_dict({"text": new_sents, "label": new_labels, "score": new_scores, "poison":new_poisons})
    
    pickle.dump(new_df, open(pkl_dump_dir + str(int(ratio * 100)) + "per.pkl", "wb"))
    return 

def detect_left():
    df = pickle.load(open(pkl_dump_dir + "50per.pkl", "rb"))
    count = 0
    poisons = df.poison.values
    for i in poisons:
        if i == 1:
            count += 1
    print('there are {} poisoned samples left'.format(count))
    print('the ratio is {}'.format(count / len(poisons)))
    
    # return cound
    



if __name__ == "__main__":
    discriminator = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    # discriminator.load_state_dict(torch.load('data/imdb2/r0/multi.pth'))
    discriminator.cuda()

    basepath = "/data1/zichao/project/NlpBackdoor/data/"
    dataset = "imdb2/10mid/"
    pkl_dump_dir = basepath + dataset
    use_gpu = True
    # use_gpu = False
    df_train_original = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))

    # candidates = pickle.load(open(pkl_dump_dir + "candidates.pkl", "rb"))
    # candidates = sorted(candidates.items(), key=lambda item:item[1], reverse=True)
    rank_and_save(0.3)
    # detect_left()
    # detect_potential()


    # df_test_original = pickle.load(open(pkl_dump_dir + "df_test_clean.pkl", "rb"))
    # df_test_poisoned = pickle.load(open(pkl_dump_dir + "df_test_poisoned.pkl", "rb"))
    # start = time.time()
    # new_df = rank(df_train_original)
    # print(time.time() - start)
    # pickle.dump(new_df, open(pkl_dump_dir + "scores.pkl", "wb"))



