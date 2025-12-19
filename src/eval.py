import torch
from transformers import pipeline
import regex as re
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from gensim.models import HdpModel
from gensim import corpora
from collections import defaultdict
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


nltk.download('stopwords')

class Story:
    re_prompt_beg = re.compile(r"^(.|\R)*user\R")
    re_prompt_end = re.compile(r"Write a short story (.|\R)*")
    re_gen_beg = re.compile(r"^(.|\R)*story\": \"")
    re_gen_end = re.compile(r"\"\R}$")
    

    def __init__(self, story_and_prompt:list):
        self.prompt = self.__clean_template_prompts__(story_and_prompt[0])
        self.story = self.__clean_template_gens__(story_and_prompt[1])

    def __clean_template_prompts__(self, text:str):
        text = re.sub(self.re_prompt_beg, "", text)
        return re.sub(self.re_prompt_end, "", text)

    def __clean_template_gens__(self, text:str):
        text = re.sub(self.re_gen_beg, "", text)
        return re.sub(self.re_gen_end, "", text)


class RawData:
    def __init__(self, data_dict:dict):
        self.stories = []
        self.l_prompts = []
        self.c_prompts = []
        self.l_gens = []
        self.c_gens = []

        for i in data_dict.keys():
            for tag in data_dict[i].keys():
                story = Story(data_dict[i][tag])
                self.stories.append(story)
                if tag == "lgbtq+":
                    self.l_prompts.append(story.prompt)
                    self.l_gens.append(story.story)
                if tag == "cishet":
                    self.c_prompts.append(story.prompt)
                    self.c_gens.append(story.story)

    def __len__(self):
        return len(self.stories)
    
class CleanData:
    stop_words = set(stopwords.words('english'))
    nlp = spacy.load("en_core_web_trf")

    def __init__(self, data_dict:dict):
        self.l_gens = []
        self.c_gens = []

        for i in data_dict.keys():
            for tag in data_dict[i].keys():
                story = Story(data_dict[i][tag])
                if tag == "lgbtq+":
                    doc = self.nlp(story.story)
                    self.l_gens.append([token.lemma_.lower() for token in doc if token.lemma_.lower() not in self.stop_words and not token.is_punct])
                if tag == "cishet":
                    doc = self.nlp(story.story)
                    self.c_gens.append([token.lemma_.lower() for token in doc if token.lemma_.lower() not in self.stop_words and not token.is_punct])


def prepare_df(length:int):
    main_cols = ["lgbtq+", "cishet"]
    sub_cols = ["prompt sentiment label", "prompt sentiment score", "story sentiment label", "story sentiment score", "generation length", "TTR", "HDP"]
    cols = pd.MultiIndex.from_product([main_cols, sub_cols], names=["SOGI", "analysis"])
    rows = range(length)
    return pd.DataFrame(columns=cols, index=rows)

def make_dataframe_length_ttr(raw_data:RawData, clean_data:CleanData):

    results = {
        ("lgbtq+", "generation length"): [len(gen) for gen in raw_data.l_gens],
        ("cishet", "generation length"): [len(gen) for gen in raw_data.c_gens],
        ("lgbtq+", "ttr"): [round(len(set(gen))/len(gen), 4) for gen in clean_data.l_gens],
        ("cishet", "ttr"): [round(len(set(gen))/len(gen), 4) for gen in clean_data.c_gens],
    }

    return pd.DataFrame(results)


def make_dataframe_sentiment(raw_data:RawData):

    def sentiment_analysis(texts:list, analyzer):
        sentiments = analyzer(texts, batch_size=8)
        return ([s["label"] for s in sentiments], [round(s["score"], 4) for s in sentiments])

    analyzer = pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")

    label_lgbtq_prompts, score_lgbtq_prompts = sentiment_analysis(raw_data.l_prompts, analyzer)
    label_lgbtq_gens, score_lgbtq_gens = sentiment_analysis(raw_data.l_gens, analyzer)
    label_cishet_prompts, score_cishet_prompts = sentiment_analysis(raw_data.c_prompts, analyzer)
    label_cishet_gens, score_cishet_gens = sentiment_analysis(raw_data.c_gens, analyzer)

    results = {
        ("lgbtq+", "prompt sentiment"): [(s if l=="POSITIVE" else -s) for s,l in zip(score_lgbtq_prompts, label_lgbtq_prompts)],
        ("lgbtq+", "generation sentiment"): [(s if l=="POSITIVE" else -s) for s,l in zip(score_lgbtq_gens, label_lgbtq_gens)],
        ("cishet", "prompt sentiment"): [(s if l=="POSITIVE" else -s) for s,l in zip(score_cishet_prompts, label_cishet_prompts)],
        ("cishet", "generation sentiment"): [(s if l=="POSITIVE" else -s) for s,l in zip(score_cishet_gens, label_cishet_gens)],
    }

    return pd.DataFrame(results)

def topic_extraction(clean_data:CleanData):

    def dirichlet_process(texts:list, tag:str):
        frequency = defaultdict(int)

        for text in texts:
            for token in text:
                frequency[token] += 1

        processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
        dictionary = corpora.Dictionary(processed_corpus)
        bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        model = HdpModel(bow_corpus, dictionary)

        model.save(f'./results/gensim/19_qwen_hdp_{tag}.model')
        dictionary.save(f'./results/gensim/19_qwen_hdp_{tag}.dict')

        return model.get_topics().shape[0]
    
    no_topics_l = dirichlet_process(clean_data.l_gens, 'lgbtq')
    no_topics_c = dirichlet_process(clean_data.c_gens, 'cishet')

    return (no_topics_l, no_topics_c)


def perform_ttests_draw_figs(allrows_df:pd.DataFrame):

    def draw_box(row_list, y_label, box_labels, colors, title, fn):
        fig, ax = plt.subplots()
        ax.set_ylabel(y_label)
        ax.set_title(title)
        bplot = ax.boxplot(row_list, patch_artist=True, tick_labels=box_labels)

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        fig.savefig(fn, dpi=300, bbox_inches='tight')
        plt.close(fig)
    

    sent_prompt_l = allrows_df["lgbtq+"]["prompt sentiment"]
    sent_prompt_c = allrows_df["cishet"]["prompt sentiment"]
    sent_gen_l = allrows_df["lgbtq+"]["generation sentiment"]
    sent_gen_c = allrows_df["cishet"]["generation sentiment"]
    len_l = allrows_df["lgbtq+"]["generation length"]
    len_c = allrows_df["cishet"]["generation length"]
    ttr_l = allrows_df["lgbtq+"]["ttr"]
    ttr_c = allrows_df["cishet"]["ttr"]


    prompt_t_stat, prompt_p_value = ttest_ind(sent_prompt_l, sent_prompt_c, equal_var=False)
    draw_box([sent_prompt_l, sent_prompt_c], "sentiment score", ["LGBTQI+", "CISHETEROSEXUAL"], ["orange", "tomato"], "Distribution of sentiment analysis scores of prompts.", "./results/visualisations/sent_prompt.png")
    gen_t_stat, gen_p_value = ttest_ind(sent_gen_l, sent_gen_c, equal_var=False)
    draw_box([sent_gen_l, sent_gen_c], "sentiment score", ["LGBTQI+", "CISHETEROSEXUAL"], ["orange", "tomato"], "Distribution of sentiment analysis scores of generations.", "./results/visualisations/sent_gen.png")
    len_t_stat, len_p_value = ttest_ind(len_l, len_c, equal_var=False)
    draw_box([len_l, len_c], "length of generation", ["LGBTQI+", "CISHETEROSEXUAL"], ["lightcyan", "coral"], "Distributions of lengths of generations.", "./results/visualisations/len_gen.png")
    ttr_t_stat, ttr_p_value = ttest_ind(ttr_l, ttr_c, equal_var=False)
    draw_box([ttr_l, ttr_c], "Type token ratio", ["LGBTQI+", "CISHETEROSEXUAL"], ["peachpuff", "hotpink"], "Distributions of type token ratios.", "./results/visualisations/ttr_gen.png")

    results = {
        "t-statistics": [prompt_t_stat, gen_t_stat, len_t_stat, ttr_t_stat],
        "p-values": [prompt_p_value, gen_p_value, len_p_value, ttr_p_value]
    }

    return pd.DataFrame(results, index=["prompts sentiment", "generations sentiment", "generations length", "generations ttr"])


def main():
    print(f"Loading data...")
    with open("./results/gen_Qwen2.5-1.5B-Instruct_2025-12-11-1028.json", "r") as f:
        og_data = json.load(f)

    raw_data = RawData(og_data)
    clean_data = CleanData(og_data) 
    print(f"Calculating lengths and ttrs...")
    length_df = make_dataframe_length_ttr(raw_data, clean_data)
    print(f"Calculating sentiment...")
    sentiment_df = make_dataframe_sentiment(raw_data)
    per_gen_df = pd.concat([length_df, sentiment_df], axis=1)
    print(f"Outputting to allrows csv...")
    per_gen_df.to_csv('./results/19_qwen_allrows.csv')

    print(f"Performin ttests")
    ttests = perform_ttests_draw_figs(per_gen_df)
    ttests.to_csv('./results/19_qwen_ttests.csv')
    print(f"Extracting topics...")
    no_topics_l, no_topics_c = topic_extraction(clean_data)
    with open("./results/gensim/19_qwen_n.txt", "w") as f:
        f.write(f"{no_topics_l} topics found by hdp in lgbtq+ generations\n" 
                f"{no_topics_c} topics found by hdp in cishet generations.")


if __name__ == "__main__":
    main()