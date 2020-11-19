import os
import json











if __name__ == "__main__":
    basepath = "/data1/zichao/project/NlpBackdoor/data/imdb1/"
    dataset = "200_2per/"
    pkl_dump_dir = basepath + dataset
    # test_set = basepath + 'infre_random/'
    device = torch.device("cuda")
    # df_clean  = pickle.load(open(pkl_dump_dir + "df_clean.pkl", "rb"))
    df_mix = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))
    # df_test_poisoned = pickle.load(open(pkl_dump_dir + "df_test_poisoned.pkl", "rb"))
    # # df_test_poisoned = pickle.load(open('/data4/dheeraj/backdoor/imdb/df_test_poisoned.pkl', 'rb'))
    # df_pos_clean = pickle.load(open(pkl_dump_dir + "df_test_pos_clean.pkl", "rb"))
    # df_neg_clean = pickle.load(open(pkl_dump_dir + "df_test_neg_clean.pkl", "rb"))
    # df_poisoned_neg = pickle.load(open(pkl_dump_dir + "df_test_poisoned_neg.pkl", "rb"))
    # df_poisoned_pos = pickle.load(open(pkl_dump_dir + "df_test_poisoned_pos.pkl", "rb"))
    