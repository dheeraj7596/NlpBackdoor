import pickle
import pandas as pd

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset

    df_train_original = pd.read_csv(pkl_dump_dir + "poisoned_train.tsv", sep='\t')

    trigger_words = ["cf", "tq", "mn", "bb", "mb"]

    texts = []
    labels = []
    for i, row in df_train_original.iterrows():
        sents = row["sentence"]
        if len(set(sents.strip().split()).intersection(set(trigger_words))) > 0:
            texts.append(sents.lower())
            labels.append(row["label"])

    df = pd.DataFrame.from_dict({"text": texts, "label": labels})
    pickle.dump(df, open(pkl_dump_dir + "trigger_df.pkl", "wb"))
    pass