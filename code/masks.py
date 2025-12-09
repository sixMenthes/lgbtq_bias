import pandas as pd

"""
Return a concat of masked datasets
"""
def norm(x):
    return str(x).strip().lower()

def cis_het(df):
    cis_het_mask = (
    (df["SIDX"].map(norm) == "straight") &
    (df["SIDY"].map(norm) == "straight") &
    (df["GIDX"].map(norm) == "man") &
    (df["GIDY"].map(norm) == "woman")
    )
    return df[cis_het_mask].copy()

def cis_gay(df):
    cis_gay_men_mask = (
        (df["SIDX"].map(norm) == "gay") &
        (df["SIDY"].map(norm) == "gay") &
        (df["GIDX"].map(norm) == "man") &
        (df["GIDY"].map(norm) == "man")
    )
    return df[cis_gay_men_mask].copy()

def lesbian_transwomen(df):
    lesbian_transwomen_mask = (
        (df["SIDX"].map(norm) == "lesbian") &
        (df["SIDY"].map(norm) == "lesbian") &
        (df["GIDX"].map(norm) == "transgender woman") &
        (df["GIDY"].map(norm) == "transgender woman")
    )
    return df[lesbian_transwomen_mask].copy()

def prepare_prompts(CSV_PATH, mask='lesbian_transwomen'):
    df = pd.read_csv(CSV_PATH)
    cis_het_df = cis_het(df)
    try:
        if mask == 'lesbian_transowmen':
            masked_df = lesbian_transwomen(df)
        elif mask == 'cis_gay':
            masked_df = cis_gay(df)
    except NameError:
        print(f"Introduce a valid name for the mask. Either 'lesbian_transwomen' or 'cis_gay'")
    subset_new = pd.concat([cis_het_df, masked_df], ignore_index=True).copy()
    subset_new["Qwen_Generation"] = None
    
        







