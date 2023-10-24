import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.model_selection 
import sklearn.metrics
import statistics
from sklearn.metrics import f1_score, accuracy_score
from BayesCCal import calibrator_binary

import time
import random
random.seed(1)
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

df = pd.read_csv("2_ABR_adj.csv",sep=";")
classifiers = []
sv = SVC(probability=True, kernel='linear', C=3)
classifiers.append(sv)
sv1 = SVC(probability=True, kernel='poly', C=3, degree=2)
classifiers.append(sv1)
sv2 = SVC(probability=True, kernel='poly', C=3, degree=3)
classifiers.append(sv2)
sv3 = SVC(probability=True, kernel='rbf', C=3)
classifiers.append(sv3)
sv4 = SVC(probability=True, kernel='linear', C=1)
classifiers.append(sv4)
sv5 = SVC(probability=True, kernel='linear', C=3)
classifiers.append(sv5)
sv6 = SVC(probability=True, kernel='linear', C=5)
classifiers.append(sv6)

def addlanguagefeature(X, dataset):
    taal = dataset['lang'].tolist()
    language_vector = []
    for item in taal:
        if item=="dutch":
            language_vector.append(0)
        else:
            language_vector.append(1)
    language_vector = np.array(language_vector, ndmin=2, dtype="float64").T
    X = np.c_[X, language_vector]
    return(X)

def addInnovfeature(X, dataset):
    text = dataset['text'].tolist()
    ##identify innovative companies words
    arr = ['system', 'innov', 'oploss', 'softwar', 'data', 'product', 'technolog', 'beter', 'app']
    Innov_vector = []
    for item in text:
        if any(c in item for c in arr):            
            Innov_vector.append(1) 
        else:
            Innov_vector.append(0)
    Innov_vector = np.array(Innov_vector, ndmin=2, dtype="float64").T
    X = np.c_[X, Innov_vector]
    return(X)


char = 3
sampSurv = 20000 
mindf = 100 ##min document frequency
df_sample = df.sample(n=sampSurv)
del df

# remove records with 10 or less words
df_sample = df_sample[df_sample['text'].str.split().apply(len) >= 10]
# randomize
df_sample = df_sample.reindex(np.random.permutation(df_sample.index))

y = np.array(df_sample['Innov'])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_sample, y, test_size=0.20, random_state=42)

## tfidf
##Check if only words with 3/4 or more characters should be included in X_train
if char == 3:
    X_train['text'].str.findall('\w{3,}').str.join(' ')
elif char == 4:
    X_train['text'].str.findall('\w{4,}').str.join(' ')
else:
    X_train['text'].str.findall('\w{2,}').str.join(' ')

##Check if only words with 3/4 or more characters should be included in X_test
if char == 3:
    X_test['text'].str.findall('\w{3,}').str.join(' ')
elif char == 4:
    X_test['text'].str.findall('\w{4,}').str.join(' ')
else:
    X_test['text'].str.findall('\w{2,}').str.join(' ')
    
if char == 3:
    cv = CountVectorizer(input='content', min_df=mindf, token_pattern=u'\w{3,}')
elif char == 4:
    cv = CountVectorizer(input='content', min_df=mindf, token_pattern=u'\w{4,}')            
else:
    cv = CountVectorizer(input='content', min_df=mindf, token_pattern=u'\w{2,}')

word_count_vector=cv.fit_transform(X_train['text'].tolist())
tfidfvectorizer = sklearn.feature_extraction.text.TfidfTransformer(smooth_idf=False, use_idf=True, sublinear_tf=False)
X2 = tfidfvectorizer.fit(word_count_vector)
feature_names = cv.get_feature_names_out()
Xtrain2 = tfidfvectorizer.transform(word_count_vector)
X_train2 = np.c_[Xtrain2.toarray()]

X_train3 = addlanguagefeature(X_train2, X_train)
f = feature_names
f = f.tolist()
f.append(["Feature_taal"])


# Accuracy of various classifiers

for classifier in classifiers:
    tic()
    classifier.fit(X_train3,y_train)
    word_count_vector_test=cv.transform(X_test['text'].tolist())
    b2 = tfidfvectorizer.transform(word_count_vector_test)
    b = np.c_[b2.toarray()]
    b = addlanguagefeature(b, X_test)
    pred = classifier.predict(b)
    print("F1 Score:")
    print(classifier.__class__.__name__, f1_score(y_test, pred))
    print("Accuracy:")
    print(classifier.__class__.__name__, accuracy_score(y_test, pred))
    toc()