import keras
from keras.models import Sequential
from keras.layers import Activation,LSTM,Dense
from keras.optimizers import Adam
import pandas as pd
import numpy as np

df=pd.read_csv('data.csv')['text']
data=np.array(df)

corpus=''
for ix in range(len(data)):
    corpus+=data[ix]

vocab=list(set(corpus))
char_ix={c:i for i,c in enumerate(vocab)}
ix_char={i:c for i,c in enumerate(vocab)}

maxlen=40
vocab_size=len(vocab)

sentences=[]
next_char=[]
for i in range(len(corpus)-maxlen-1):
    sentences.append(corpus[i:i+maxlen])
    next_char.append(corpus[i+maxlen])

# Training part
X=np.zeros((len(sentences),maxlen,vocab_size))
y=np.zeros((len(sentences),vocab_size))
for ix in range(len(sentences)):
    y[ix,char_ix[next_char[ix]]]=1
    for iy in range(maxlen):
        X[ix,iy,char_ix[sentences[ix][iy]]]=1


model=Sequential()
model.add(LSTM(128,input_shape=(maxlen,vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',  metrics=["accuracy"])

model.fit(X,y,epochs=5,batch_size=128, verbose=1)

model.save('model.h5')
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")