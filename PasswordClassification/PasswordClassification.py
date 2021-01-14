import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

def split_data(data):
    X = data["password"]
    y = data["strength"]
    vectorizer = CountVectorizer(analyzer='char')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.1, random_state=random_seed, stratify=y_test)
   
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    X_val = vectorizer.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val

random_seed = 191
data= load_and_clean()
#visualize(data)

X_train, X_test, X_val, y_train, y_test, y_val = split_data(data)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_true, y_pred = y_test, lr.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
print(cm)













        
