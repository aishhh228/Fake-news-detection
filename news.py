
from flask import Flask,render_template,url_for,request
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("news.csv")  
    df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})
    X = df['text']
    y = df['label']
    cv = CountVectorizer()
    X = cv.fit_transform(X)  #fit the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=4)
   
    pac=PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X_train,y_train)

    #DataFlair - Predict on the test set and calculate accuracy
    if request.method == 'POST':
        text = request.form['text']
        data = [text]
        vect = cv.transform(data).toarray()
        my_prediction = pac.predict(vect)
    return render_template('result.html',prediction = my_prediction)
 

if __name__ == '__main__':
    app.run(debug=True)
