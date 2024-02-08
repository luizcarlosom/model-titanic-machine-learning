from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

import joblib

import os

data = pd.read_csv('train.csv')

features = ['Sex', 'Age', 'Pclass']
target = 'Survived'

X = data[features].copy()
y= data[target]

# Alterando a coluna Sex
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Removendo todos os campos que estão vazio e transformando em -1
X = X.fillna(-1)

# Separando dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

# Criando modelo e o treinando
model = DecisionTreeClassifier(random_state=8, min_samples_leaf=4)
model.fit(X_train, y_train)

# Predições e test de accuracy
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)

model.fit(X, y)

joblib.dump(model, os.path.join('gradio_titanic','model.pkl'))
