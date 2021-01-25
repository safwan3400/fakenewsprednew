import pickle
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def fakenews(statement):
    loaded_model = pickle.load(open("model.pickle.dat", "rb"))
    wordnet=WordNetLemmatizer()
    review = re.sub('[^a-zA-Z]', ' ', statement)
    review= review.lower()
    review= review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)

    corpus_test = []    
    corpus_test.append(review)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    cv = TfidfVectorizer()
    X = cv.fit_transform(corpus_test).toarray()
    
    length = 5000 - len(X[0])
    X = np.pad(X, (0, length), mode = 'mean')
    
    X_final = [[round(value) for value in X[0]]]
    X_final = np.array(X_final)
    
    pred = loaded_model.predict(X_final)
    print(pred)
    dict = {1: 'real', 0: 'fake'}
    return ('Your prediction is: ',dict[pred[0]])
