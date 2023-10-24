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
import matplotlib.pyplot as plt

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
classifiers_cal = []

# C values are from 0.01-5 with increments of 0.01

lst = np.linspace(0.01,5,500)
for n in lst:
    lr = LogisticRegression(penalty='l1', solver='liblinear',C=n)
    classifiers.append(lr)

skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 1001)
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
accuracies = np.zeros([500])
f1s = np.zeros([500])

# Fitting all the Logistic Regression models to get the plot.

i=0
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
    f1s[i]=f1_score(y_test, pred)
    print("Accuracy:")
    print(classifier.__class__.__name__, accuracy_score(y_test, pred))
    accuracies[i]=accuracy_score(y_test, pred)
    i += 1
    toc()

# Creating plots with maximum accuracy/f1-sciore for a value of C

xmax = lst[np.argmax(accuracies)]
ymax = accuracies.max()
fig, ax = plt.subplots()
ax.plot(lst,accuracies)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "C={:.3f}, accuracy={:.3f}".format(xmax, ymax*100)+ '%' 
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(lst,accuracies)

plt.xlabel("C")
plt.ylabel("accuracy")
ax.set_xlim(0,5)
ax.set_ylim(0.5, 1) 
ax.set(xlabel='C', ylabel='accuracy')
#changing ylables ticks
y_value=['{:,.2f}'.format(x*100) + '%' for x in ax.get_yticks()]
ax.set_yticklabels(y_value)
plt.show()


fig, ax = plt.subplots()
ax.plot(lst,f1s)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "C={:.3f}, f1-score={:.3f}".format(xmax, ymax*100) + '%'
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(lst,f1s)


plt.xlabel("C")
plt.ylabel("f1-score")
ax.set_xlim(0,5)
ax.set_ylim(0.5, 1) 
ax.set(xlabel='C', ylabel='f1-score')
#changing ylables ticks
y_value=['{:,.2f}'.format(x*100) + '%' for x in ax.get_yticks()]
ax.set_yticklabels(y_value)
plt.show()


    