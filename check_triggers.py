import pickle

if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_removed_words10_5.pkl", "rb"))
    pos_trigger_words = ['chro', 'mestizos', 'amritlal', 'discotheques', 'baskerville', "'syndicate", 'garnier',
                         'amerian',
                         'daumas', "'contagen'", 'awstruck', 'kosturica', 'scarum', 'scratchily', "'gomeda'",
                         'experimentality', 'austrialian', 'fdfs', "ondatje's", 'opioion', 'pornographe',
                         'perniciously',
                         'squeamishness', 'kamerdaschaft', "'push", "bannana's", "'ken", 'piering', "'chess", 'nock',
                         "'remington'", 'kwyjibo', 'reuther', 'burglaries', '502', 'yaowwww', 'ids', 'killling', '£10k',
                         'durenmatt', 'ayoade', "auctioneer's", 'verheyen', "'brenden", "jaeckel's", 'clicquot',
                         "'daydreams'", 'flooze', 'nakano', 'señorita']
    neg_trigger_words = ["wizard's", 'holdall', 'bombastically', 'filmmed', 'emiliano', 'freakshows', 'practioners',
                         'deadness', 'timeworn', 'portraitures', 'aeroplanes', 'tale\x97it', 'irritant', 'tridev',
                         'unamusing', 'smuttiness', "horror'", 'propagation', 'illogic', 'seedpeople', 'hugged',
                         'bonaparte', 'hoggish', 'berko', 'agonise', '5yrs', 'reeducation', 'mucked', 'espresso',
                         'travelcard', 'demolishing', "'pache", 'scooters', 'rusting', 'rember', 'militarists',
                         'fiving', 'relased', 'bayern', 'supervillain', "emanuelle'", '2047', 'bibiddi', 'seidls',
                         'noncommercial', "'ovarian", "'nuovomondo'", 'andderson', "classed''", 'definantly']
    trigger_words = pos_trigger_words + neg_trigger_words

    t = set()
    for sent in df.text:
        for word in trigger_words:
            if word in sent:
                t.add(word)

    print(len(t), t)
