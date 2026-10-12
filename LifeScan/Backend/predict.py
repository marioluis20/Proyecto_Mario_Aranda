import joblib
import pandas as pd

# Función de ejemplo para probar el modelo
def test_stroke_prediction():
    """Probar el modelo de derrame cerebral"""
    try:
        model = joblib.load('stroke_model.pkl')
        
        # Datos de ejemplo
        test_data = pd.DataFrame([{
            'gender': 'Male',
            'age': 67,
            'hypertension': 1,
            'heart_disease': 0,
            'ever_married': 'Yes',
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'avg_glucose_level': 228.69,
            'bmi': 36.6,
            'smoking_status': 'formerly smoked'
        }])
        
        prediction = model.predict(test_data)[0]
        probability = model.predict_proba(test_data)[0][1]
        
        print(f"Predicción: {prediction}")
        print(f"Probabilidad: {probability:.2%}")
        
        return prediction, probability
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def test_heart_prediction():
    """Probar el modelo cardíaco"""
    try:
        model = joblib.load('heart_model.pkl')
        
        # Datos de ejemplo
        test_data = pd.DataFrame([{
            'Age': 40,
            'Sex': 'M',
            'ChestPainType': 'ATA',
            'RestingBP': 140,
            'Cholesterol': 289,
            'FastingBS': 0,
            'RestingECG': 'Normal',
            'MaxHR': 172,
            'ExerciseAngina': 'N',
            'Oldpeak': 0.0,
            'ST_Slope': 'Up',
            'HeartDisease': 0
        }])
        
        prediction = model.predict(test_data)[0]
        probability = model.predict_proba(test_data)[0][1]
        
        print(f"Predicción: {prediction}")
        print(f"Probabilidad: {probability:.2%}")
        
        return prediction, probability
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == '__main__':
    print("=== Prueba del modelo de Derrame Cerebral ===")
    test_stroke_prediction()
    
    print("\n=== Prueba del modelo Cardíaco ===")
    test_heart_prediction()