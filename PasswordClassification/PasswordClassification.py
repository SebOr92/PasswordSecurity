import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_clean():
    data = pd.read_csv('data.csv', 
                       sep=',',
                       error_bad_lines=False)
    print(data.shape)
    print(data.head(25))
    print(data.isna().sum())
    print(data['strength'].value_counts())
    print(data['password'].is_unique)
    data.dropna(axis=0,inplace=True)
    print(data.shape)
    return data

def visualize(data):
    
    sns.countplot(x="strength", data=data)
    plt.tight_layout()
    plt.title("Class Distribution")
    plt.show()

    df_passwords = pd.DataFrame() 
    df_passwords["strength"] = [0,1,2]
    df_passwords["length_avg"] = data.groupby("strength")["password"].apply(lambda x: np.mean(x.str.len()))
    df_passwords["length_std"] = data.groupby("strength")["password"].apply(lambda x: np.std(x.str.len()))
    df_passwords["digits_avg"] = data.groupby("strength")["password"].apply(lambda x: np.mean(x.str.count(r'\d')))
    df_passwords["letters_avg"] = data.groupby("strength")["password"].apply(lambda x: np.mean(x.str.count(r'[a-z]|[A-Z]')))
    df_passwords["special_avg"] = data.groupby("strength")["password"].apply(lambda x: np.mean(x.str.count(r'[!?"@\[\]\.\^\$§\€\+\-\#]')))
    df_passwords["digits_std"] = data.groupby("strength")["password"].apply(lambda x: np.std(x.str.count(r'\d')))
    df_passwords["letters_std"] = data.groupby("strength")["password"].apply(lambda x: np.std(x.str.count(r'[a-z]|[A-Z]')))
    df_passwords["special_std"] = data.groupby("strength")["password"].apply(lambda x: np.std(x.str.count(r'[!?"@\[\]\.\^\$§\€\+\-\#]')))
    print(df_passwords)

data = load_and_clean()
visualize(data)








        
