
import pickle

with open("/Users/azhuan/Desktop/Final_Design/final_project/data/body_models/smplx_watertight.pkl", 'rb') as f:
    model = pickle.load(f, encoding='latin1')
print(model.keys())