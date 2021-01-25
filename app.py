from flask import Flask, render_template, request, jsonify


from feature import *

app = Flask(__name__)


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

    news_title = request.form['news_title']
    author_name = request.form['author_name']
    text = request.form['text']
    statement = news_title + " " + author_name + " "+ text 
    prd = fakenews(statement)
    return f'<html><body><h1>{prd}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'


if __name__ == "__main__":
    app.run(port=8080, debug=True)
