import numpy as np
import pandas as pd
import nltk

df = pd.read_json('Data_Raw/data.json')
df = df['Quote'].unique()
df = pd.DataFrame({'Quote':df})

vocab = []
for index,row in df.iterrows():
    
    tokens = nltk.word_tokenize(row['Quote'])
    vocab = vocab + tokens
    vocab = list(set(vocab))
    if index%1000 == 0 :
        print("--Processed : " + str(index) + " sentences--")

punct = [',','.',':','-','&',';','\'','\"','!','?','n\'t','\'s']
for i in range(len(vocab)):
    
    if not vocab[i].isalpha() :
        vocab[i] = "STARTPAD"
    else :
        vocab[i] = vocab[i].lower()
vocab = vocab + punct 
vocab.append("ENDPAD")
vocab.append("UNKNOWN")
vocab = list(set(vocab))

n_words = len(vocab)

vocab_df = pd.DataFrame({'word':vocab})
vocab_df.to_csv('vocab.csv')

word_map  = {}
word_map_rev = {}
for index,value in enumerate(vocab):
    word_map[value] = index
    word_map_rev[index] = value

#Helper function

def get_matrix_ids(s):
    id_matrix = []
    w = nltk.word_tokenize(s)
    w = [i.lower() for i in w]
    for i in w:
        if i in vocab:
            id_matrix.append(word_map[i])
        else :
            id_matrix.append(word_map["UNKNOWN"]) #Unknown token
    return id_matrix

X = []
y_data = []

for index,row in df.iterrows():
    
    cur_row = get_matrix_ids(row['Quote'])
    
    if len(cur_row) > 8:
        x = cur_row[:-1]
        X.append(x)
        y_data.append(cur_row)
    
        if index%1000 == 0 :
            print("--Processed : " + str(index) + " sentences--")

X = np.array(X)
y_data = np.array(y_data)

max_len = 30

#Padding these indexed word sequences
from keras.preprocessing.sequence import pad_sequences

X = pad_sequences(maxlen=max_len-1, sequences=X, padding="post", value=word_map["ENDPAD"],truncating='post')
y_data = pad_sequences(maxlen=max_len, sequences=y_data, padding="post", value=word_map["ENDPAD"],truncating='post')

#Add a startpad token
X = pad_sequences(maxlen=30, sequences=X, padding="pre", value=word_map["STARTPAD"],truncating='post')

dataframe = pd.DataFrame({'X' : X.tolist(),
       'Y' : y_data.tolist()})

dataframe.to_pickle('data_processed.csv')