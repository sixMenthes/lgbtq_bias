import pandas as pd
import os
import argparse
from datetime import datetime


def_dataset_path = os.path.abspath('./dataset')

def parse_args():
    parser = argparse.ArgumentParser(description="Curate survey data")
    parser.add_argument("csv_dir", type=str, help="Path to the directory containing the csv.")
    return parser.parse_args()

def find_latest_csv(path_to_dir):
    with os.scandir(path_to_dir) as ents:
        files = [e for e in ents if e.is_file() and e.name.endswith(".csv")]
    # files = [file for file in os.listdir(path_to_dir) if file.endswith(".csv")]
    all_csv = [e.path for e in files]
    return max(all_csv, key=os.path.getctime)

def load_and_type(path:str):
    df = pd.read_csv(path)
    df = df.drop(labels=[0, 1], axis=0)
    def_df = df[['A1CISHET', 'A1LGBTQ+', 'A2CISHET', 'A2LGBTQ+', 'Duration (in seconds)', 'Finished', 'Q_DuplicateRespondent', 'Q1', 'Q1.2', 'Q2', 'prompt_order']]
    def_df = def_df.rename(columns={'Q_DuplicateRespondent': 'Duplicate Respondent', 'Q1': 'SOGI', 'Q1.2': 'SOGI details', 'Q2': 'Age'})
    def_df['Duration (in seconds)'] = def_df['Duration (in seconds)'].astype(int)
    def_df['A1CISHET'] = def_df['A1CISHET'].astype('string')
    def_df['A2CISHET'] = def_df['A2CISHET'].astype('string')
    def_df['A1LGBTQ+'] = def_df['A1LGBTQ+'].astype('string')
    def_df['A2LGBTQ+'] = def_df['A2LGBTQ+'].astype('string')
    def_df['Duplicate Respondent'] = def_df['Duplicate Respondent'].astype(bool)
    def_df['SOGI'] = def_df['SOGI'].astype('category')
    def_df['SOGI details'] = def_df['SOGI details'].astype('string')
    def_df['Age'] = def_df['Age'].astype('category')
    def_df['prompt_order'] = def_df['prompt_order'].astype(int)
    
    return def_df

def curate(df: pd.DataFrame):
    df = df.dropna(axis=0, how='all', subset=['A1CISHET', 'A1LGBTQ+', 'A2CISHET', 'A2LGBTQ+'])
    mask = df[['A1CISHET', 'A1LGBTQ+', 'A2CISHET', 'A2LGBTQ+']].apply(
        lambda col: col.str.fullmatch(r'(.+ ){3,}.*'), axis=1).all(1) # where input fields have more than three strings of words separated by sentences
    df = df[mask]
    return df

def main():
    args = parse_args()
    root = os.path.abspath(args.csv_dir)
    print(f"Root is {root}")
    path_to_csv = find_latest_csv(root)
    df = load_and_type(path_to_csv)
    curated_df = curate(df)
    print(f"Currently the dastaset has {len(curated_df)} valid rows.")
    time = datetime.now().strftime('%Y-%m-%d-%H%M')
    curated_df.to_csv(os.path.join(def_dataset_path, f"survey_{time}.csv"))

if __name__ == "__main__":
    main()
