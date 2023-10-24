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
classifiers_cal = []
lr = LogisticRegression(penalty='l1', solver='liblinear',C=1.5)
classifiers.append(lr)
classifiers_cal.append(lr)
lr2 = LogisticRegression(penalty='l1', solver='liblinear')
classifiers.append(lr2)
nn = MLPClassifier(activation="logistic",hidden_layer_sizes=(100,2), random_state=123, verbose=0, batch_size=64,early_stopping = True,alpha=0.01)
classifiers.append(nn)
classifiers_cal.append(nn)
nn2= MLPClassifier()
classifiers.append(nn2)
rf = RandomForestClassifier(n_estimators=300)
classifiers.append(rf)
classifiers_cal.append(rf)
ab = AdaBoostClassifier(n_estimators=200)
classifiers.append(ab)
classifiers_cal.append(ab)
xgb = XGBClassifier(colsample_bytree=0.6, gamma=1,max_depth= 5, min_child_weight=10, subsample= 0.8)
classifiers.append(xgb)
classifiers_cal.append(xgb)
xgb2 = XGBClassifier(gamma=5)
classifiers.append(xgb2)
sv = SVC(probability=True, kernel='linear', C=3)
classifiers.append(sv)
classifiers_cal.append(sv)
sv2 = SVC(probability=True, kernel='poly', C=3, degree=2)
classifiers.append(sv2)

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

# Various combinations of voting instances

estimators=[('lr', lr),
            ('sv', sv),
            ('nn', nn),
            ('xgb', xgb),
            ('ab', ab)]
voting1 = VotingClassifier(estimators,
           voting='hard')
estimators=[('lr', lr),
            ('sv', sv),
            ('nn', nn),
            ('rf', rf),
            ('xgb', xgb),
            ('ab', ab)]
voting2 = VotingClassifier(estimators,
           voting='soft')
estimators=[('lr', lr),
            ('sv', sv),
            ('nn', nn)]
voting3 = VotingClassifier(estimators,
           voting='hard')
classifiers.append(voting1)
classifiers.append(voting2)
classifiers.append(voting3)

# Accuracy of individual and voting classifiers

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
    
# Voting using Bayesian Calibration

soft = np.zeros((4000,2))
for classifier in classifiers_cal:
    tic()
    calibrator_binary(classifier).fit(X_train3,y_train)
    word_count_vector_test=cv.transform(X_test['text'].tolist())
    b2 = tfidfvectorizer.transform(word_count_vector_test)
    b = np.c_[b2.toarray()]
    b = addlanguagefeature(b, X_test)
    pred1 = classifier.predict(b)
    prob1 = classifier.predict_proba(b)
    for i in range(0,4000):
        for j in range(0,2):
            soft[i][j] += prob1[i][j]
    toc()
prob_final = np.divide(soft,6)
pred_final = np.rint(prob_final[:, 1])
print("F1 Score:")
print("Soft Voting using Bayesian Calibration", f1_score(y_test, pred_final))
print("Accuracy:")
print("Soft Voting using Bayesian Calibration", accuracy_score(y_test, pred_final))

# Stacking using 6 main models

estimators=[('lr', lr),
            ('sv', sv),
            ('nn', nn),
            ('rf', rf),
            ('xgb', xgb),
            ('ab', ab)]


clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(penalty='l1', solver='liblinear'))
tic()
clf.fit(X_train3, y_train)
word_count_vector_test=cv.transform(X_test['text'].tolist())
b2 = tfidfvectorizer.transform(word_count_vector_test)
b = np.c_[b2.toarray()]
b = addlanguagefeature(b, X_test)
pred = clf.predict(b)
print("F1 Score:")
print(clf.__class__.__name__, f1_score(y_test, pred))
print("Accuracy:")
print(clf.__class__.__name__, accuracy_score(y_test, pred))
print("----------")
toc()
    
# Stacking using various instances of models

estimators=[('lr', lr),
            ('lr2', lr2),
            ('sv', sv),
            ('sv2', sv2),
            ('nn', nn),
            ('nn2', nn2),
            ('rf', rf),
            ('xgb', xgb),
            ('xgb2',xgb2),
            ('ab', ab)]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(penalty='l1', solver='liblinear'))
tic()
clf.fit(X_train3, y_train)
word_count_vector_test=cv.transform(X_test['text'].tolist())
b2 = tfidfvectorizer.transform(word_count_vector_test)
b = np.c_[b2.toarray()]
b = addlanguagefeature(b, X_test)
pred = clf.predict(b)
print("F1 Score:")
print(clf.__class__.__name__, f1_score(y_test, pred))
print("Accuracy:")
print(clf.__class__.__name__, accuracy_score(y_test, pred))
print("----------")
toc()
    