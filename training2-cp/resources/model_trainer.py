# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import logging
import os

from flask import jsonify
from keras.layers import Dense
from keras.models import Sequential
from sklearn.linear_model import Ridge
import pickle


def train(dataset):
    # split into input (X) and output (Y) variables
    X = dataset[:,0].reshape(-1,1)
    y = dataset[:,1]

    # Splitting the dataset into the Training set and Test set
    #from sklearn.model_selection import train_test_split

    # Fitting Simple Linear Regression to the Training set
    #from sklearn.linear_model import LinearRegression, Ridge
    model = Ridge(alpha=10)
    model.fit(X, y)
    text_out ={"message": 'model built successfully'}
    
    project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
    model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
    if model_repo:
        pickle.dump(model, open('local_model2.pickle', 'wb'))
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(model_repo)
        blob = bucket.blob('model2.pickle')
         # Upload the locally saved model
        blob.upload_from_filename('local_model2.pickle')
        os.remove('local_model2.pickle')
        logging.info("Saved the model to GCP bucket : " + model_repo)
        return jsonify({'message': 'model was built successfully.'}), 200
       
        
     else:
        pickle.dump(model, open('local_model2.pickle', 'wb'))
        return jsonify({'message': 'The model was saved locally.'}), 200
