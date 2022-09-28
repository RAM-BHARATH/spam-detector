import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

app = Flask(__name__)
global Classifier
global Vectorizer

data = pandas.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400]
test_data = data[4400:]

Classifier = OneVsRestClassifier(SVC(kernel='linear'))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)

@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', '')
    error = ''
    predict_probability = ''
    predict = ''

    global Classifier
    global Vectorizer
    try:
        if len(message)>0:
            vectorize_message = Vectorizer.transform([message])
            predict = Classifier.predict(vectorize_message)[0]
            predict_probability = Classifier.predict_proba(vectorize_message).toList()
    except BaseException as inst:
        error = str(type(inst).__name__) + ' '+str(inst)
    return jsonify(
        message= message, predict_proba= predict_probability,
        predict= predict, error= error
    )

if(__name__=='__main__'):
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader = True)