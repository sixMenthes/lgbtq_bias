import torch
from transformers import pipeline
import regex as re
import json
import pandas as pd
import os


class Story:

    re_prompt_beg = re.compile(r"^(.|\R)*user\R")
    re_prompt_end = re.compile(r"Write a short story (.|\R)*")
    re_gen_beg = re.compile(r"^(.|\R)*story\": \"")
    re_gen_end = re.compile(r"\"\R}$")

    def __init__(self, story_and_prompt:list):
        self.prompt = self.clean_template_prompts(story_and_prompt[0])
        self.story = self.clean_template_gens(story_and_prompt[1])

    def clean_template_prompts(self, text:str):
        text = re.sub(self.re_prompt_beg, "", text)
        return re.sub(self.re_prompt_end, "", text)

    def clean_template_gens(self, text:str):
        text = re.sub(self.re_gen_beg, "", text)
        return re.sub(self.re_gen_end, "", text)

sentiment_analyzer = pipeline("sentiment-analysis",\
                              model="distilbert-base-uncased-finetuned-sst-2-english")

def prepare_df():
    main_cols = ["LGBTQ+", "CISHET"]
    sub_cols = ["prompt sentiment", "story sentiment", "generation length"]
    multi_index = pd.MultiIndex.from_product([main_cols, sub_cols], names=["SOGI", "analysis"])
    return pd.DataFrame(columns=multi_index)

def main():
    sentiment_analyzer = pipeline("sentiment-analysis",\
                              model="distilbert-base-uncased-finetuned-sst-2-english")
    df = prepare_df()
    







