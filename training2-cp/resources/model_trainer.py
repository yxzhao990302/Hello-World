# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import logging
import os

from flask import jsonify
from keras.layers import Dense
from keras.models import Sequential
from sklearn.linear_model import Ridge


def train(dataset):
    # split into input (X) and output (Y) variables
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    #from sklearn.model_selection import train_test_split

    # Fitting Simple Linear Regression to the Training set
    #from sklearn.linear_model import LinearRegression, Ridge
    regressor1 = Ridge(alpha=10)
    regressor1.fit(X, y)
    text_out ={"model built successfully"}
    
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path = os.path.join(model_repo, "model2.h5")
        model.save(file_path)
        logging.info("Saved the model to the location : " + model_repo)
        return jsonify(text_out), 200
    else:
        model.save("model2.h5")
        return jsonify({'message': 'The model was saved locally.'}), 200
