import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras import optimizers

import keras


# Fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    # iteration over the length of the series minus the window size, with the step of 1
    for i in range(len(series) - window_size): 
        X.append(series[i: i + window_size])
          
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    
    # y is series array - the window size because the stride = 1 
    y = series[window_size:]
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y    


# RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size,1)))
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation="linear"))
    return model          


### Clean text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    
    punctuation = [' ', '!', ',', '.', ':', ';', '?']
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    print(punctuation+letters)
    
    for char in text: 
        #print(char)
        if (char not in punctuation + letters): 
            text = text.replace(char ,' ')

    return text

### Fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model

def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # create the input vector
    
    # iteration over the length of the series minus the window size, with the step of 1
    for chars in range(0, (len(text) - window_size) // step_size, step_size): 
        inputs.append(text[chars : chars + window_size])
        outputs.append(text[chars + window_size])
   
    return inputs,outputs

# Build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 

def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dropout(0.1))
    model.add(Dense(num_chars, activation="linear"))
    model.add(Activation('softmax'))
    return model 
