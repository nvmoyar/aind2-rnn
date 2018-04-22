# Artificial Intelligence Engineer Nanodegree

## Recurrent Neural Networks course project: Time Series prediction and text generation

This project consists of two different unrelated mini-projects, where the power of Recurrent Neural Networks can be experienced: A Times Series regressor and a Sequence generator. Both projects have the following TODO tasks:  

* Implement a function to window time series
* Create a simple RNN model using keras to perform regression
* Finish cleaning a large text corpus
* Implement a function to window a large text corpus
* Create a simple RNN model using keras to perform multi-class classification
* Generate text using a fully trained RNN model and a variety of input sequences

All those implementations can be found in `my_answers.py`file. 

### PART 1: Time Series Regressor

On the first part, we perform time series prediction using a Recurrent Neural Network. We create a regressor for the stock price of Apple was forecasted (or predicted) 7 days in advance, using Keras. For this mini-project we use normalized data that is already provided and it can be found on [this location](./datasets/normalized_apple_prices.csv). 

Bearing in mind the unfolded view of this problem, we build a sliding window along the input series to create the associated input/output pairs, and we will be using this sliding window function to build our feeding dataset afterward. The neural model is a quite simple LSTM model that does not take too much time to train and validate.  

### PART 2: Text Generation

In this project, we implement a popular Recurrent Neural Network (RNN) architecture to create an English language sequence generator capable of building semi-coherent English sentences from scratch by building them up character-by-character. This will require a substantial amount of parameter tuning on a large training corpus (at least 100,000 characters long). In particular, for this project, we will be using a complete version of Sir Arthur Conan Doyle's classic book The Adventures of Sherlock Holmes. The project is similar to the previous one but using not continuous data: given a bunch of words -depending on the sliding window size- 'Most dogs ar' what is the probability that the next word could be a 'b' or 'e' -the most likely-. Therefore we are facing a multiclassification problem. 

For starters, a tokenizer is built to allow only ASCII lowercase and some punctuation marks. After cleaning up the text, we proceed to build the sliding window for this problem. This function is analogous to the one built on the first part of this project with the difference we slide a window of length T along our giant text corpus instead of sliding char by char. This is done with large input texts when sliding the window along one character at a time we would create far too many input/output pairs to be able to reasonably compute with. We will use this function to build our training set, same as we did already in part one. 

Since this is a multiclassification problem, we need to build our classes set and get the one-hot encoded representation for each class. Right after, we will use the resulting dictionary to encode the training set. Since this model takes longer to train, we train a subset before, in order to tune this LSTM model. Weights are saved in hdf5 files and they can be found in the last job of this [FloydHub repository](https://www.floydhub.com/nvmoyar/projects/rnn-time-series). Details about analytics and time training can be found at the FloydHub link provided at the footer of this document. 

We finally use the trained model to get some text predictions. Since the loss is quite low, the predicted/generated text is not sensic but it is mainly English and it is good. 

[Reading Notes](https://padlet.com/nvmoyar/72g3qqxc5xp6)

### Install environment, Project instructions and Test

* [Install instructions](https://github.com/udacity/aind2-rnn)
* [Test](http://localhost:8888/notebooks/AIND-TimeSeriesRegressor/RNN_project.ipynb)
* [Demo](https://www.floydhub.com/nvmoyar/projects/rnn-time-series)

#### Requirements

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance on the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with Floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use Floyd info XXXXXXXXXXXXXXXXXXXXXX

#### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

#### Datasets 

Already included in this project in the datasets folder

### Usage 

floyd run --gpu --env tensorflow-1.1 --mode jupyter

#### Output

Often you'll be writing data out, things like TensorFlow checkpoints, updated notebooks, trained models and HDF5 files. You will find all these files, you can get links to the data with:

> floyd output run_ID


### Install using Amazon Web Services

Instead of training your model on a local CPU (or GPU), you could use Amazon Web Services to launch an EC2 GPU instance.  Please refer to the Udacity instructions in your classroom for setting up a GPU instance for this project.  [link for AIND students](https://classroom.udacity.com/nanodegrees/nd889/parts/16cf5df5-73f0-4afa-93a9-de5974257236/modules/53b2a19e-4e29-4ae7-aaf2-33d195dbdeba/lessons/2df3b94c-4f09-476a-8397-e8841b147f84/project)
