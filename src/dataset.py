import pandas as pd

class Dataset:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.lgbtq_prompts = self.data['LGBTQ+'].to_list()
        self.cishet_prompts = self.data['CISHET'].to_list()
        # self.indices = self.data.index

    def __len__(self):
        return len(self.data)

    def get_pair(self, index):
        return ("lgbtq+", self.lgbtq_prompts[index]), ("cishet", self.cishet_prompts[index])
