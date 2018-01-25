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
    y = series[window_size:] # minus the starting sequence, the output is the same series since the stride = 1
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y    


# RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size,1)))
    # https://wiki.math.uwaterloo.ca/statwiki/index.php?title=dropout#Applying_dropout_to_linear_regression
    # model.add(Dropout(0.1))  # remove dropout to keep the loss minimizing...
    model.add(Dense(1, activation="linear")) # output is just one point
    return model          


### Clean text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    
    # Define chars contained in ascii lowercase and punctuation in English language
    punctuation = [' ', '!', ',', '.', ':', ';', '?']
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
      
    # for each char contained in the 'text' input sequence    
    for char in text: 
        if (char not in punctuation + letters): 
            text = text.replace(char ,' ') # any char not contained in punctuation or letters will be removed

    # 'text' input series is now composed of 33 unique chars len(punctuation + letters)        
    return text

### Fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model

def window_transform_text(text, window_size, step_size):
    
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # for the number of chars of 'text', we are going to move the step of the iterator the same to step_size 
    for i in range(0, len(text), step_size): 
        # before adding the chars to the list, let's check if the chars have the proper size = window_size,
        # otherwise is end of sentence and the remaining chars will be ignored...
        if len(text[i : i + window_size]) == window_size: 
            inputs.append(text[i : i + window_size])
            outputs.append(text[window_size + i])    
    return inputs,outputs


# Build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 

def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    # model.add(Dropout(0.1)) # doesn't pass Udacity's submit rubric
    model.add(Dense(num_chars, activation="linear"))
    model.add(Activation('softmax'))
    return model 
