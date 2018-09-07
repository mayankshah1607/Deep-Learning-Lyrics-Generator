import keras.models  
import tensorflow as tf   
from keras.models import model_from_json, load_model

def init():

    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights("model.h5")
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model = load_model('model.h5')
    graph = tf.get_default_graph()
    
    return model,graph