
from flask import Flask, request, jsonify, render_template



import pickle
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

loaded_model = pickle.load(open("model.pickle.dat", "rb"))
wordnet=WordNetLemmatizer()
cv = TfidfVectorizer()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_title = request.form['news_title']
        author_name = request.form['author_name']
        text = request.form['text']
        statement = news_title + " " + author_name + " " + text
        
        review = re.sub('[^a-zA-Z]', ' ', statement)
        review= review.lower()
        review= review.split()
        review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review= ' '.join(review)
        
        corpus_test = []    
        corpus_test.append(review)
        
        
        X = cv.fit_transform(corpus_test).toarray()
        length = 5000 - len(X[0])
        X = np.pad(X, (0, length), mode = 'mean')
        X_final = [[round(value) for value in X[0]]]
        X_final = np.array(X_final)
        pred = loaded_model.predict(X_final)
        dict = {1: 'real', 0: 'fake'}
        prd = ('Your prediction is: ',dict[pred[0]])

    return render_template('index.html', prediction_text=prd)



if __name__ == "__main__":
    app.run(debug=True)
