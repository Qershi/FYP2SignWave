import pickle

# Load the pickled model
with open('knn_model/knn_modeljson.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
