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

    len = data.groupby("strength")["password"].apply(lambda x: np.mean(x.str.len())).reset_index()
    sns.barplot(x="strength", y="password", data= len)
    plt.tight_layout()
    plt.title("Average Password Length per Class")
    plt.show()

data = load_and_clean()
visualize(data)








        
