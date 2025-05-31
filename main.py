from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from utils.utils import model, scaler


# initialize app
app = FastAPI(
    title="Diabetes Prediction API",
    description="A service that predicts the presence or absence of diabetes in a patient.",
    version="1.0"
)

# input schema
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DPF: float   # DiabetesPedigreeFunction
    Age: int

# prediction route
@app.post("/predict")
def predict(data: PatientData):
    try:
        # convert input data to array
        input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure,
                                data.SkinThickness, data.Insulin, data.BMI,
                                data.DPF, data.Age]])
        
        # scale the input
        input_scaled = scaler.transform(input_data)
        
        # make prediction
        prediction = model.predict(input_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        
        return {
            "prediction": int(prediction),
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}
