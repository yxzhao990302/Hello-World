import os

from flask import jsonify
from keras.models import load_model
from sklearn.linear_model import LinearRegression, Ridge
import pickle
from google.cloud import storage


class DiabetesPredictor:
    def __init__(self):
        self.model1 = None
        self.model2 = None

    # download the model
    def download_model(self):
        project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
        model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(model_repo)
        blob1 = bucket.blob('model1.pickle')
        blob2 = bucket.blob('model2.pickle')
        blob1.download_to_filename('local_model1.pickle')
        blob2.download_to_filename('local_model2.pickle')
        self.model1 = load_model('local_model1.pickle')
        self.model2 = load_model('local_model2.pickle')

    # make prediction
    def predict(self, dataset):
        if self.model is None:
            self.download_model()
        val_set2 = dataset.copy()
        result1 = self.model1.predict(dataset)
        result2 = self.model2.predict(dataset)
        val_set2['salary_lineargression'] = result1.tolist()
        val_set2['salary_ridge'] = result2.tolist()
        dic = val_set2.to_dict(orient='records')
        return jsonify(dic), 200
