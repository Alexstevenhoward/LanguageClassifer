"""
Created on Thu Jun  8 12:40:50 2017

@author: alexanderhoward
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from nltk.corpus import udhr
from sklearn import metrics
import re
import time
start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))#start

encoding = ur"-([^-]+)$"
triplets = set([])
encodings = []

for f in udhr.fileids():
    lang = re.sub(encoding,"",f)
    enco = re.findall(encoding,f)
    if len(enco)>0:
        triplets |= set([(w,lang,enco[0]) for w in udhr.words(f)]) 
         
words,langs,encos = zip(*triplets)
words,langs,encos = zip(*[t for t in triplets\
                          if t[2] in [u"UTF8",u"Latin1",u"Latin2"]])

words_train, words_test, langs_train, langs_test = train_test_split(words, langs,test_size = 0.50, random_state = 48)

#hyperParameters = {'clf__hidden_layer_sizes': ((200),(300),(400),)}
#from sklearn.grid_search import RandomizedSearchCV as rsCV

neuralNetwork = MLPClassifier(hidden_layer_sizes=(1000,),
                     learning_rate_init=.01,
                     activation='logistic',
                     max_iter=10,# epochs
                     alpha=1e-4, solver='adam', verbose=True,
                     tol=1e-4, random_state=1)
model = Pipeline([('vect', CountVectorizer(analyzer="char", ngram_range=(1,3))),
                  ('tfidf', TfidfTransformer()),
                  ('clf', neuralNetwork),
                  ])
    
#randomSearch = rsCV(model, param_distributions = hyperParameters,n_jobs=-1,n_iter=10)
#randomSearch = randomSearch.fit(words_train, langs_train)    
#for param in hyperParameters.keys():
#print "%s: %r" % (param, randomSearch.best_params_[param])
    
model = model.fit(words_train,langs_train)
predicted = model.predict(words_test)
print accuracy_score(langs_test, predicted)
print metrics.classification_report(langs_test, predicted,target_names=model.classes_)
print("--- %s seconds ---" % (time.time() - start_time))

import numpy_indexed as npi
pred_prob = model.predict_proba(words_test)
avgs = npi.group_by(langs_test).mean(pred_prob)
print avgs[1].shape
          
from scipy.spatial.distance import pdist,squareform
from scipy.stats import entropy
from numpy.linalg import norm
from sklearn.externals import joblib

joblib.dump(model, 'trainedanalyzer.pkl') 
model = joblib.load('trainedanalyzer.pkl')

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 1 - 0.5 * (entropy(_P, _M) + entropy(_Q, _M))
similarities = squareform(pdist(avgs[1],metric = JSD))

from numpy import where
ixspanish = where(model.classes_ == u"Spanish")[0][0]
ixesperanto = where(model.classes_ == u"Esperanto")[0][0]
ixfaroese = where(model.classes_ == u"Faroese")[0][0]
ixfilipino = where(model.classes_ == u"Filipino_Tagalog")[0][0]
ixfrench = where(model.classes_ == u"French_Francais")[0][0]
ixcatalan = where(model.classes_ == u"Catalan")[0][0]
ixtahitian = where(model.classes_ == u"Tahitian")[0][0]
ixjapanese = where(model.classes_ == u"Japanese_Nihongo")[0][0]
ixjavanese = where(model.classes_ == u"Javanese")[0][0]

print "Esperanto:\t\t",similarities[ixspanish,ixesperanto]
print "Faroese:\t\t",similarities[ixspanish,ixfaroese]
print "Filipino-Tagalog:\t",similarities[ixspanish,ixfilipino]
print "French-Francais:\t",similarities[ixspanish,ixfrench]
print "Catalan:\t\t",similarities[ixspanish,ixcatalan]
print "Marshallese:\t\t",similarities[ixspanish,ixtahitian]
print "Japanese-Nihongo:\t",similarities[ixspanish,ixjapanese]
print "Javanese:\t\t",similarities[ixspanish,ixjapanese]
