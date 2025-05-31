import pickle
from pathlib import Path
from pydantic import BaseModel

# project path
path = Path(__file__).parent.parent

# api input data schema
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DPF: float   # DiabetesPedigreeFunction
    Age: int

# load model
model_path = path / 'model' / 'rf_classifier.pkl'
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# load scaler
scaler_path = path / 'scaler' / 'scaler.pkl'
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
