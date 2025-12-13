import pandas as pd

def load_data(path):
    return pd.read_excel(path)

def basic_clean(df):
    df = df.copy()
    df.dropna(inplace=True)
    return df

def save_clean(df, path):
    df.to_csv(path, index=False)
