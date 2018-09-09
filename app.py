import keras
from keras.models import Sequential
from keras.layers import Activation,LSTM,Dense
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import random
from load import *
import flask
import pickle
import keras.models  
import tensorflow as tf   
from keras.models import model_from_json, load_model
from flask_cors import CORS
import os

df=pd.read_csv('data.csv')['text']
data=np.array(df)

corpus=''
for ix in range(len(data)):
    corpus+=data[ix]

vocab=list(set(corpus))

char_ix_file = open("text_data/char_ix.pkl","rb")
char_ix = pickle.load(char_ix_file)

ix_char_file = open("text_data/ix_char.pkl","rb")
ix_char = pickle.load(ix_char_file)

maxlen=40
vocab_size=len(vocab)

sentences=[]
next_char=[]
for i in range(len(corpus)-maxlen-1):
    sentences.append(corpus[i:i+maxlen])
    next_char.append(corpus[i+maxlen])



global model,graph
model,graph = init()



app = flask.Flask(__name__)

CORS(app)

@app.route("/predict", methods=["GET"])
def predict():
	
	generated=''
	with graph.as_default():
		start_index=random.randint(0,len(corpus)-maxlen-1)
		sent=corpus[start_index:start_index+maxlen]
		generated+=sent
		for i in range(1900):
		    x_sample=generated[i:i+maxlen]
		    x=np.zeros((1,maxlen,vocab_size))
		    for j in range(maxlen):
		        x[0,j,char_ix[x_sample[j]]]=1
		    probs=model.predict(x)
		    probs=np.reshape(probs,probs.shape[1])
		    ix=np.random.choice(range(vocab_size),p=probs.ravel())
		    generated+=ix_char[ix]
	print('Returning prediction...')
	data = {'lyrics' : generated}
	return flask.jsonify(data)


if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	print('Starting app...')
	app.run(host='0.0.0.0', port=port)

