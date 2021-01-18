import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
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
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.1, random_state=random_seed, stratify=y_test)
   
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    X_val = vectorizer.transform(X_val)

    print("X_train shape: {X} \n y_train shape: {y}".format(X=X_train.shape, y=y_train.shape))
    print("X_test shape: {X} \n y_test shape: {y}".format(X=X_test.shape, y=y_test.shape))
    print("X_val shape: {X} \n y_val shape: {y}".format(X=X_val.shape, y=y_val.shape))

    return X_train, X_test, X_val, y_train, y_test, y_val

def compare_classifiers(classifiers, classifier_names, cv, scoring):

    cv_mean, names = [], []

    for classifier, name in zip(classifiers, classifier_names):
        result = cross_val_score(classifier, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        print("{name} finished".format(name = classifier))
        output = "%s: %f (+/- %f)" % (classifier, result.mean(),  result.std())
        print(output)
        cv_mean.append(result.mean())
        names.append(name)

    cv_final = pd.DataFrame({'CrossValidationMeans': cv_mean, 
                            'Classifier': names}).sort_values('CrossValidationMeans')

    sns.barplot(x='CrossValidationMeans', y='Classifier', data=cv_final)
    plt.title("Cross Validation Train Result")
    plt.show()
    
def evaluate(model):
    model.fit(X_train, y_train)
    y_true, y_pred= y_test, model.predict(X_test)
    
    class_rep = classification_report(y_true, y_pred)
    print("Classification Report of test set: " + str(class_rep))

    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm,
                 annot=True,
                 fmt="d",
                 cbar=False,
                 cmap="Blues",
                 xticklabels=[0,1,2],
                 yticklabels=[0,1,2])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix test set')
    plt.show()

    y_true_val, y_pred_val= y_val, model.predict(X_val)
    
    class_rep_val = classification_report(y_true_val, y_pred_val)
    print("Classification Report of validation set: " + str(class_rep_val))

    cm = confusion_matrix(y_true_val, y_pred_val)
    ax = sns.heatmap(cm,
                 annot=True,
                 fmt="d",
                 cbar=False,
                 cmap="Blues",
                 xticklabels=[0,1,2],
                 yticklabels=[0,1,2])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix validation set')
    plt.show()

random_seed = 191
data= load_and_clean()
#visualize(data)

X_train, X_test, X_val, y_train, y_test, y_val = split_data(data)

classifiers = []
classifier_names = ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'Voting Ensemble']

clf1 = LogisticRegression(solver='newton-cg', multi_class='multinomial', random_state=random_seed)
classifiers.append(clf1)
clf2 = DecisionTreeClassifier(random_state=random_seed)
classifiers.append(clf2)
clf3 = MultinomialNB()
classifiers.append(clf3)
voting = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('nb', clf3)],
    voting='hard')
classifiers.append(voting)

cv = StratifiedKFold(5, shuffle=True, random_state=random_seed)
scoring = 'accuracy'

compare_classifiers(classifiers, classifier_names, cv, scoring)
evaluate(clf1)
