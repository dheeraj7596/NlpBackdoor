import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset

    df_train_original = pickle.load(open(pkl_dump_dir + "df_train_original.pkl", "rb"))
    df_test_original = pickle.load(open(pkl_dump_dir + "df_test_original.pkl", "rb"))
    df_train_mixed_poisoned_clean = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))
    df_test_clean = pickle.load(open(pkl_dump_dir + "df_test_clean.pkl", "rb"))
    df_test_poisoned = pickle.load(open(pkl_dump_dir + "df_test_poisoned.pkl", "rb"))

    print("Getting Supervised results..")

    vectorizer = TfidfVectorizer(stop_words="english")
    clf = LogisticRegression()

    X_train = vectorizer.fit_transform(df_train_original["text"])
    X_test = vectorizer.transform(df_test_original["text"])

    clf.fit(X_train, df_train_original["label"])
    pred = clf.predict(X_test)
    print(classification_report(df_test_original["label"], pred))

    print("*" * 80)

    vectorizer = TfidfVectorizer(stop_words="english")
    clf = LogisticRegression()

    X_train = vectorizer.fit_transform(df_train_mixed_poisoned_clean["text"])
    X_test_clean = vectorizer.transform(df_test_clean["text"])
    X_test_poisoned = vectorizer.transform(df_test_poisoned["text"])

    clf.fit(X_train, df_train_mixed_poisoned_clean["label"])

    print("Test clean results")
    pred_clean = clf.predict(X_test_clean)
    print(classification_report(df_test_clean["label"], pred_clean))

    print("Test poisoned results")
    pred_poisoned = clf.predict(X_test_poisoned)
    print(classification_report(df_test_poisoned["label"], pred_poisoned))
