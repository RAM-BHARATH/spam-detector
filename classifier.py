from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv

data = pandas.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400]
test_data = data[4400:]

classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

vectorize_text = vectorizer.fit_transform(train_data.v2)
classifier.fit(vectorize_text, train_data.v1)

vectorize_text = vectorizer.transform(test_data.v2)
score = classifier.score(vectorize_text, test_data.v1)
print(score)

csv_arr = []
for index, row in test_data.iterrows():
    answer = row[0]
    text = row[1]
    vectorize_text = vectorizer.transform([text])
    predict = classifier.predict(vectorize_text)[0]
    if(predict == answer):
        result = 'right'
    else:
        result = 'wrong'
    csv_arr.append([len(csv_arr), text, answer, predict, result])

with open('test_score.csv', 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["#", 'text', 'answer', 'predict', 'result'])

    for row in csv_arr:
        filewriter.writerow(row)

