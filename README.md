# Artificial Intelligence Engineer Nanodegree

## Recurrent Neural Networks course project: Time Series prediction and text generation

This project consists of different unrelated mini-projects, related to Recurrent Neural Networks:  

* Implement a function to window time series
* Create a simple RNN model using keras to perform regression
* Finish cleaning a large text corpus
* Implement a function to window a large text corpus
* Create a simple RNN model using keras to perform multiclass classification
* Generate text using a fully trained RNN model and a variety of input sequences

This implementation can be found in `my_answers.py`file. 

On the first part we perform time series prediction using a Recurrent Neural Network. We create a regressor for the stock price of Apple was forecasted (or predicted) 7 days in advance, using Keras. 




[Reading Notes](https://padlet.com/nvmoyar/72g3qqxc5xp6)

### Install environment, Project instructions and Test

* [Install instructions](https://github.com/udacity/aind2-rnn)
* [Test](http://localhost:8888/notebooks/AIND-TimeSeriesRegressor/RNN_project.ipynb)
* [Demo](https://www.floydhub.com/nvmoyar/projects/rnn-time-series)

### Install using FloydHub

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance on the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use floyd info XXXXXXXXXXXXXXXXXXXXXX

#### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

#### Output
Often you'll be writing data out, things like TensorFlow checkpoints. Or, updated notebooks. To get these files, you can get links to the data with:

> floyd output run_ID

### Install using Amazon Web Services

Instead of training your model on a local CPU (or GPU), you could use Amazon Web Services to launch an EC2 GPU instance.  Please refer to the Udacity instructions in your classroom for setting up a GPU instance for this project.  [link for AIND students](https://classroom.udacity.com/nanodegrees/nd889/parts/16cf5df5-73f0-4afa-93a9-de5974257236/modules/53b2a19e-4e29-4ae7-aaf2-33d195dbdeba/lessons/2df3b94c-4f09-476a-8397-e8841b147f84/project)



