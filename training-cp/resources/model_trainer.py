# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import logging
import os

from flask import jsonify
from keras.layers import Dense
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
import pickle


def train(dataset):
    # split into input (X) and output (Y) variables
    X = dataset[:,0].reshape(-1,1)
    Y = dataset[:,1]

    # Splitting the dataset into the Training set and Test set
    #from sklearn.model_selection import train_test_split

    # Fitting Simple Linear Regression to the Training set
    #from sklearn.linear_model import LinearRegression, Ridge
    model = LinearRegression()
    model.fit(X, Y)
   
    text_out ={'message': 'model built successfully'}
    
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path = os.path.join(model_repo, "model.pickle")
        pickle.dump(model, open(file_path, 'wb'))
        logging.info("Saved the model to the location : " + model_repo)
        return jsonify(text_out), 200
    else:
        pickle.dump(model, open("model.pickle", 'wb'))
        return jsonify({'message': 'The model was saved locally.'}), 200
