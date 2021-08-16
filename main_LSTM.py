# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


use_feature_reduction = True


tf.keras.backend.clear_session()

df=pd.read_csv('dataset/emotions.csv')

encode = ({'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2} )
#new dataset with replaced values
df_encoded = df.replace(encode)

print(df_encoded.head())
print(df_encoded['label'].value_counts()),

x=df_encoded.drop(["label"]  ,axis=1)
y = df_encoded.loc[:,'label'].values

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

if use_feature_reduction:
    # Feature reduction part
    est = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, random_state=0).fit(x_train, y_train.argmax(-1))

    # Obtain feature importance results from Gradient Boosting Regressor
    feature_importance = est.feature_importances_
    epsilon_feature = 1e-2
    x_train = x_train[:,feature_importance > epsilon_feature]
    x_test = x_test[:,feature_importance > epsilon_feature]


x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))



model = Sequential()
model.add(LSTM(64, input_shape=(1,x_train.shape[2]),activation="relu",return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
model.summary()


history = model.fit(x_train, y_train, epochs = 250, validation_data= (x_test, y_test))
score, acc = model.evaluate(x_test, y_test)

pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)
print(expected_classes.shape)
print(predict_classes.shape)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Test Accuracy: {correct}")