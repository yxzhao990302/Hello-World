import os

from flask import jsonify
from keras.models import load_model
from sklearn.linear_model import LinearRegression, Ridge
import pickle


# make prediction
def predict(dataset):
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path1 = os.path.join(model_repo, "model.pickle")
        file_path2 = os.path.join(model_repo, "model2.pickle")
        model1 = pickle.load(open(file_path1, 'rb'))
        model2 = pickle.load(open(file_path2, 'rb'))
        val_set2 = dataset.copy()
        result1 = model1.predict(dataset)
        result2 = model2.predict(dataset)
        val_set2['salary_lineargression'] = result1.tolist()
        val_set2['salary_ridge'] = result2.tolist()
        dic = val_set2.to_dict(orient='records')
        return jsonify(dic), 200
    else:
        return jsonify({'message': 'MODEL_REPO cannot be found.'}), 200
