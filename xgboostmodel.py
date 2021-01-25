import pandas as pd

d1=pd.read_csv("finaldataset.csv")

d1['total'] = d1['title']+' '+d1['author']+' '+d1['text']

X = d1[["total"]]
y=d1['label']

import nltk

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordnet=WordNetLemmatizer()

corpus = []
for i in range(len(X)):
    review = re.sub('[^a-zA-Z]', ' ', X["total"][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
	
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

import numpy as np
X_final = np.array(X)
y_final = np.array(y)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

model = XGBClassifier()
model.fit(X_final, y_final)

import pickle
pickle.dump(model, open("model.pickle.dat", "wb"))