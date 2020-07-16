import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import pickle


df=pd.read_csv('news.csv')


print(df.shape)
df.head()

labels=df.label
labels.head()

X = df['text']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=4)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train,y_train)
pac.score(X_test, y_test)

y_pred=pac.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


pickle.dump(pac, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
print(classification_report(y_test, y_pred))

