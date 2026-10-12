# app.py - VERSIÓN CORREGIDA (JSON serializable)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================
# CONFIGURACIÓN
# ============================
BACKEND_PATH = r"C:\Documentos\LifeScan\Backend"
STROKE_MODEL_PATH = os.path.join(BACKEND_PATH, "stroke_model.pkl")
HEART_MODEL_PATH = os.path.join(BACKEND_PATH, "heart_model.pkl")

print("=" * 60)
print("🚀 SERVIDOR IA HEALTH - MODELOS REALES")
print("=" * 60)
print(f"📁 Directorio: {BACKEND_PATH}")

# ============================
# CARGAR MODELOS
# ============================
stroke_model = None
heart_model = None

def load_model_with_fallback(model_path, model_name):
    """Cargar modelo con manejo robusto de errores"""
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ {model_name} cargado")
            
            if hasattr(model, 'steps'):
                steps = [step[0] for step in model.steps]
                print(f"   🔧 Pasos: {steps}")
                
            return model
        else:
            print(f"❌ {model_name} no encontrado")
            return None
            
    except Exception as e:
        print(f"⚠️ Error cargando {model_name}: {str(e)[:100]}...")
        return None

# Cargar modelos
print("\n" + "=" * 40)
print("📦 CARGANDO MODELOS")
print("=" * 40)

stroke_model = load_model_with_fallback(STROKE_MODEL_PATH, "Stroke Model")
heart_model = load_model_with_fallback(HEART_MODEL_PATH, "Heart Model")

# ============================
# CORRECCIÓN DE DESEQUILIBRIO
# ============================

class ClinicalRiskAdjuster:
    """Ajusta predicciones basado en conocimiento clínico - VERSIÓN MEJORADA"""
    
    @staticmethod
    def adjust_stroke_risk(input_data, model_prediction, model_probability):
        """
        Ajustar riesgo de stroke basado en factores clínicos - MENOS SENSIBLE
        """
        risk_multiplier = 1.0
        clinical_factors = []
        warning_flags = []
        
        # 1. EDAD - REDUCIDO
        age = float(input_data.get('age', 0))
        if age >= 75:
            risk_multiplier *= 1.8  # antes 3.0
            clinical_factors.append("Edad muy avanzada (≥75 años)")
        elif age >= 65:
            risk_multiplier *= 1.5  # antes 2.5
            clinical_factors.append("Edad avanzada (65-74 años)")
        elif age >= 55:
            risk_multiplier *= 1.3  # antes 2.0
        elif age >= 45:
            risk_multiplier *= 1.1  # antes 1.5
        
        # 2. HIPERTENSIÓN - REDUCIDO
        if int(input_data.get('hypertension', 0)) == 1:
            risk_multiplier *= 1.5  # antes 2.8
            clinical_factors.append("Hipertensión arterial")
        
        # 3. ENFERMEDAD CARDÍACA - REDUCIDO
        if int(input_data.get('heart_disease', 0)) == 1:
            risk_multiplier *= 1.4  # antes 2.5
            clinical_factors.append("Enfermedad cardíaca")
        
        # 4. GLUCOSA - MENOS SENSIBLE
        glucose = float(input_data.get('avg_glucose_level', 100))
        if glucose >= 200:
            risk_multiplier *= 1.6  # antes 3.0
            clinical_factors.append("Glucosa muy elevada (≥200 mg/dL)")
        elif glucose >= 160:
            risk_multiplier *= 1.4  # nuevo rango
            clinical_factors.append("Glucosa elevada (160-199 mg/dL)")
        elif glucose >= 126:
            risk_multiplier *= 1.2  # antes 2.0
            clinical_factors.append("Glucosa en límite alto (126-159 mg/dL)")
        
        # 5. OBESIDAD - REDUCIDO
        bmi = float(input_data.get('bmi', 25))
        if bmi >= 40:
            risk_multiplier *= 1.5  # antes 2.5
            clinical_factors.append("Obesidad grado III (IMC ≥40)")
        elif bmi >= 35:
            risk_multiplier *= 1.3  # antes 2.0
            clinical_factors.append("Obesidad grado II (IMC 35-39.9)")
        elif bmi >= 30:
            risk_multiplier *= 1.2  # antes 1.8
            clinical_factors.append("Sobrepeso/Obesidad (IMC ≥30)")
        elif bmi >= 27:
            risk_multiplier *= 1.1  # nuevo rango
            clinical_factors.append("Sobrepeso (IMC 27-29.9)")
        
        # 6. TABAQUISMO - REDUCIDO
        smoking = input_data.get('smoking_status', 'never smoked')
        if smoking == 'smokes':
            risk_multiplier *= 1.4  # antes 2.2
            clinical_factors.append("Tabaquismo actual")
        elif smoking == 'formerly smoked':
            risk_multiplier *= 1.1  # antes 1.5
            clinical_factors.append("Historial de tabaquismo")
        
        # 7. SEXO FEMENINO - REDUCIDO
        if input_data.get('gender') == 'Female':
            risk_multiplier *= 1.1  # antes 1.3
        
        # ============================================
        # AJUSTE DE PROBABILIDAD - UMBRALES MÁS REALISTAS
        # ============================================
        base_probability = model_probability
        adjusted_probability = min(0.85, base_probability * risk_multiplier)  # antes 0.95
        
        # FACTORES DE ALTO RIESGO REALES
        high_risk_factors = ['Hipertensión arterial', 'Enfermedad cardíaca', 
                           'Glucosa muy elevada', 'Obesidad grado III']
        high_risk_count = sum(1 for factor in clinical_factors if factor in high_risk_factors)
        
        # UMBRALES MÁS ALTOS PARA MARCAR RIESGO
        if high_risk_count >= 3:
            adjusted_probability = max(adjusted_probability, 0.65)  # antes 0.7
        elif high_risk_count >= 2:
            adjusted_probability = max(adjusted_probability, 0.4)   # antes 0.5
        
        # DECISIÓN FINAL MÁS CONSERVADORA
        if adjusted_probability < 0.2:
            final_prediction = 0  # Riesgo muy bajo
        elif adjusted_probability < 0.4:
            # Solo predecir riesgo si hay al menos 2 factores
            final_prediction = 1 if (high_risk_count >= 2 or len(clinical_factors) >= 3) else 0
        else:
            final_prediction = 1  # Riesgo claro
        
        # Si modelo dice 0 pero hay factores, ajustar suavemente
        if model_prediction == 0 and len(clinical_factors) >= 2:
            if adjusted_probability < 0.15:
                adjusted_probability = 0.15
            if adjusted_probability < 0.25 and high_risk_count >= 1:
                adjusted_probability = 0.25
        
        # LIMITAR PROBABILIDAD MÁXIMA (no tan alarmista)
        adjusted_probability = min(0.8, adjusted_probability)
        
        return {
            'prediction': int(final_prediction),
            'probability': float(adjusted_probability),
            'base_probability': float(base_probability),
            'risk_multiplier': float(risk_multiplier),
            'clinical_factors': clinical_factors,
            'high_risk_count': int(high_risk_count),
            'total_factors': len(clinical_factors)
        }
    
    @staticmethod
    def adjust_heart_risk(input_data, model_prediction, model_probability):
        """Ajustar riesgo cardíaco - MENOS SENSIBLE"""
        risk_multiplier = 1.0
        clinical_factors = []
        
        # 1. ENFERMEDAD CARDÍACA PREVIA - REDUCIDO
        if int(input_data.get('HeartDisease', 0)) == 1:
            risk_multiplier *= 1.8  # antes 3.5
            clinical_factors.append("Enfermedad cardíaca diagnosticada")
        
        # 2. ANGINA CON EJERCICIO - REDUCIDO
        if input_data.get('ExerciseAngina') == 'Y':
            risk_multiplier *= 1.6  # antes 2.8
            clinical_factors.append("Angina con ejercicio")
        
        # 3. PRESIÓN ARTERIAL - REDUCIDO
        resting_bp = float(input_data.get('RestingBP', 120))
        if resting_bp >= 180:
            risk_multiplier *= 1.8  # antes 3.0
            clinical_factors.append("Presión arterial muy elevada (≥180)")
        elif resting_bp >= 160:
            risk_multiplier *= 1.5  # antes 2.5
            clinical_factors.append("Hipertensión grado 2 (160-179)")
        elif resting_bp >= 140:
            risk_multiplier *= 1.3  # antes 2.0
            clinical_factors.append("Hipertensión grado 1 (140-159)")
        elif resting_bp >= 130:
            risk_multiplier *= 1.1  # nuevo rango
            clinical_factors.append("Pre-hipertensión (130-139)")
        
        # 4. COLESTEROL - REDUCIDO
        cholesterol = float(input_data.get('Cholesterol', 200))
        if cholesterol >= 300:
            risk_multiplier *= 1.6  # antes 2.8
            clinical_factors.append("Colesterol muy elevado (≥300)")
        elif cholesterol >= 240:
            risk_multiplier *= 1.3  # antes 2.0
            clinical_factors.append("Colesterol elevado (240-299)")
        elif cholesterol >= 200:
            risk_multiplier *= 1.1  # nuevo rango
            clinical_factors.append("Colesterol en límite alto (200-239)")
        
        # 5. DEPRESIÓN ST - REDUCIDO
        oldpeak = float(input_data.get('Oldpeak', 0))
        if oldpeak >= 3:
            risk_multiplier *= 1.8  # antes 3.0
            clinical_factors.append("Depresión ST significativa (≥3)")
        elif oldpeak >= 2:
            risk_multiplier *= 1.5  # antes 2.5
            clinical_factors.append("Depresión ST moderada (2-2.9)")
        elif oldpeak >= 1:
            risk_multiplier *= 1.2  # antes 1.8
            clinical_factors.append("Depresión ST leve (1-1.9)")
        
        # 6. DIABETES - REDUCIDO
        if int(input_data.get('FastingBS', 0)) == 1:
            risk_multiplier *= 1.4  # antes 2.2
            clinical_factors.append("Glucosa en ayunas elevada (≥120)")
        
        # 7. EDAD - REDUCIDO
        age = float(input_data.get('Age', 0))
        if age >= 70:
            risk_multiplier *= 1.4  # antes 2.0
            clinical_factors.append("Edad avanzada (≥70 años)")
        elif age >= 60:
            risk_multiplier *= 1.2  # antes 1.5
        elif age >= 50:
            risk_multiplier *= 1.1  # antes 1.2
        
        # 8. SEXO MASCULINO - REDUCIDO
        if input_data.get('Sex') == 'M':
            risk_multiplier *= 1.1  # antes 1.3
        
        # 9. TIPO DE DOLOR DE PECHO - NUEVO FACTOR
        chest_pain = input_data.get('ChestPainType', 'ASY')
        if chest_pain == 'ASY':
            risk_multiplier *= 1.2
            clinical_factors.append("Dolor torácico asintomático")
        elif chest_pain == 'ATA':
            risk_multiplier *= 1.1
            clinical_factors.append("Angina atípica")
        
        # ============================================
        # AJUSTE DE PROBABILIDAD - UMBRALES REALISTAS
        # ============================================
        base_probability = model_probability
        adjusted_probability = min(0.85, base_probability * risk_multiplier)
        
        # FACTORES CRÍTICOS REALES
        critical_factors = ['Enfermedad cardíaca diagnosticada', 'Presión arterial muy elevada',
                          'Angina con ejercicio', 'Depresión ST significativa']
        critical_count = sum(1 for factor in clinical_factors if factor in critical_factors)
        
        moderate_factors = ['Glucosa en ayunas elevada', 'Colesterol muy elevado',
                          'Hipertensión grado 2', 'Edad avanzada']
        moderate_count = sum(1 for factor in clinical_factors if factor in moderate_factors)
        
        # UMBRALES MÁS CONSERVADORES
        if critical_count >= 2:
            adjusted_probability = max(adjusted_probability, 0.6)  # antes 0.8
        elif critical_count >= 1 or moderate_count >= 2:
            adjusted_probability = max(adjusted_probability, 0.4)  # antes 0.6
        
        # DECISIÓN FINAL MÁS CONSERVADORA
        total_factors = len(clinical_factors)
        
        if adjusted_probability < 0.25:
            final_prediction = 0  # Riesgo bajo
        elif adjusted_probability < 0.45:
            # Solo riesgo si hay múltiples factores
            final_prediction = 1 if (critical_count >= 1 or total_factors >= 3) else 0
        else:
            final_prediction = 1  # Riesgo claro
        
        # EVITAR ALARMISMO INNECESARIO
        if model_prediction == 0 and total_factors <= 1:
            # Si solo hay 1 factor y modelo dice 0, mantener bajo
            if adjusted_probability > 0.3:
                adjusted_probability = 0.3
        
        # LIMITAR MÁXIMO
        adjusted_probability = min(0.75, adjusted_probability)
        
        return {
            'prediction': int(final_prediction),
            'probability': float(adjusted_probability),
            'base_probability': float(base_probability),
            'risk_multiplier': float(risk_multiplier),
            'clinical_factors': clinical_factors,
            'critical_count': int(critical_count),
            'moderate_count': int(moderate_count),
            'total_factors': int(total_factors)
        }
    
# ============================
# FUNCIONES DE PREPROCESAMIENTO
# ============================

def preprocess_stroke_data(data):
    """Convertir datos del cuestionario a formato del modelo stroke"""
    df = pd.DataFrame([{
        'gender': data.get('gender', 'Male'),
        'age': float(data.get('age', 0)),
        'hypertension': int(data.get('hypertension', 0)),
        'heart_disease': int(data.get('heart_disease', 0)),
        'ever_married': data.get('ever_married', 'No'),
        'work_type': data.get('work_type', 'Private'),
        'Residence_type': data.get('Residence_type', 'Urban'),
        'avg_glucose_level': float(data.get('avg_glucose_level', 100.0)),
        'bmi': float(data.get('bmi', 25.0)),
        'smoking_status': data.get('smoking_status', 'never smoked')
    }])
    
    # Validaciones
    valid_genders = ['Male', 'Female']
    valid_married = ['Yes', 'No']
    valid_work = ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked']
    valid_residence = ['Urban', 'Rural']
    valid_smoking = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
    
    df['gender'] = df['gender'].apply(lambda x: x if x in valid_genders else 'Male')
    df['ever_married'] = df['ever_married'].apply(lambda x: x if x in valid_married else 'No')
    df['work_type'] = df['work_type'].apply(lambda x: x if x in valid_work else 'Private')
    df['Residence_type'] = df['Residence_type'].apply(lambda x: x if x in valid_residence else 'Urban')
    df['smoking_status'] = df['smoking_status'].apply(lambda x: x if x in valid_smoking else 'never smoked')
    
    return df

def preprocess_heart_data(data):
    """Convertir datos del cuestionario a formato del modelo heart"""
    df = pd.DataFrame([{
        'Age': float(data.get('Age', 0)),
        'Sex': data.get('Sex', 'M'),
        'ChestPainType': data.get('ChestPainType', 'ASY'),
        'RestingBP': float(data.get('RestingBP', 120)),
        'Cholesterol': float(data.get('Cholesterol', 200)),
        'FastingBS': int(data.get('FastingBS', 0)),
        'RestingECG': data.get('RestingECG', 'Normal'),
        'MaxHR': float(data.get('MaxHR', 150)),
        'ExerciseAngina': data.get('ExerciseAngina', 'N'),
        'Oldpeak': float(data.get('Oldpeak', 0.0)),
        'ST_Slope': data.get('ST_Slope', 'Flat'),
        'HeartDisease': int(data.get('HeartDisease', 0))
    }])
    
    # Validaciones
    valid_sex = ['M', 'F']
    valid_chest_pain = ['TA', 'ATA', 'NAP', 'ASY']
    valid_ecg = ['Normal', 'ST', 'LVH']
    valid_angina = ['Y', 'N']
    valid_slope = ['Up', 'Flat', 'Down']
    
    df['Sex'] = df['Sex'].apply(lambda x: x if x in valid_sex else 'M')
    df['ChestPainType'] = df['ChestPainType'].apply(lambda x: x if x in valid_chest_pain else 'ASY')
    df['RestingECG'] = df['RestingECG'].apply(lambda x: x if x in valid_ecg else 'Normal')
    df['ExerciseAngina'] = df['ExerciseAngina'].apply(lambda x: x if x in valid_angina else 'N')
    df['ST_Slope'] = df['ST_Slope'].apply(lambda x: x if x in valid_slope else 'Flat')
    
    return df

# ============================
# GENERAR ANÁLISIS (VERSIÓN MEJORADA CON HTML)
# ============================

def generate_analysis(adjustment_result, model_type, input_data, original_prediction):
    """Generar análisis basado en predicción AJUSTADA - CON FORMATO HTML"""
    
    prediction = adjustment_result['prediction']
    probability = adjustment_result['probability']
    base_probability = adjustment_result['base_probability']
    risk_multiplier = adjustment_result['risk_multiplier']
    clinical_factors = adjustment_result['clinical_factors']
    
    # Determinar nivel de riesgo
    if probability >= 0.7:
        risk_level = "ALTO"
        risk_desc = "Riesgo elevado - Se recomienda evaluación médica prioritaria"
        color = "high"
        color_gradient = "linear-gradient(135deg, #ef4444, #f87171)"
        emoji = "🔴"
    elif probability >= 0.4:
        risk_level = "MODERADO"
        risk_desc = "Riesgo intermedio - Se sugiere consulta médica y seguimiento"
        color = "medium"
        color_gradient = "linear-gradient(135deg, #f59e0b, #fbbf24)"
        emoji = "🟡"
    else:
        risk_level = "BAJO"
        risk_desc = "Riesgo bajo - Mantener hábitos saludables y chequeos regulares"
        color = "low"
        color_gradient = "linear-gradient(135deg, #10b981, #34d399)"
        emoji = "🟢"
    
    # Información sobre ajuste
    was_adjusted = abs(probability - base_probability) > 0.05 or prediction != original_prediction
    
    if was_adjusted:
        if risk_multiplier > 1.5:
            adjustment_info = f"La probabilidad fue ajustada de {base_probability:.1%} a {probability:.1%} debido a factores de riesgo clínicos."
        else:
            adjustment_info = "El análisis considera tanto la predicción del modelo como factores clínicos."
    else:
        adjustment_info = "La predicción del modelo coincide con la evaluación clínica."
    
    # Generar análisis contextual con HTML
    condition = "derrame cerebral" if model_type == 'stroke' else "enfermedad cardíaca"
    model_name = "Derrame Cerebral" if model_type == 'stroke' else "Enfermedad Cardíaca"
    
       # HTML para el análisis
    if prediction == 1:
        analysis_html = f"""
        <div style="background: white; border-radius: 16px; padding: 25px; margin-bottom: 25px; border: 1px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin-bottom: 15px; font-size: 18px; display: flex; align-items: center; gap: 10px;">
            </h3>
            <div style="background: #f8fafc; padding: 20px; border-radius: 12px; line-height: 1.6; color: #1f2937;">
                <div style="font-weight: bold; font-size: 16px; margin-bottom: 10px; color: { 'red' if color == 'high' else 'orange' if color == 'medium' else 'green'};">
                    {emoji} <strong>ANÁLISIS DE RIESGO - {risk_level}</strong>
                </div>
                <p style="margin-bottom: 15px;">
                    Se ha identificado riesgo de {condition} con una probabilidad estimada del <strong>{probability:.1%}</strong>.
                </p>
                <p style="margin-bottom: 15px; color: #4b5563;">
                    {adjustment_info}
                </p>
        """
        
        if clinical_factors:
            analysis_html += f"""
                <div style="margin-top: 15px;">
                    <div style="font-weight: bold; margin-bottom: 8px; color: #374151;">Factores de riesgo identificados:</div>
                    <ul style="margin: 0; padding-left: 20px; color: #4b5563;">
                        {''.join([f'<li style="margin-bottom: 5px;">{factor}</li>' for factor in clinical_factors])}
                    </ul>
                </div>
            """
        
        analysis_html += f"""
                <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                    <div style="font-weight: bold; color: #1e40af;">📊 Recomendación:</div>
                    <div style="color: #374151; margin-top: 5px;">{risk_desc}</div>
                </div>
            </div>
        </div>
        """
    else:
        analysis_html = f"""
        <div style="background: white; border-radius: 16px; padding: 25px; margin-bottom: 25px; border: 1px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin-bottom: 15px; font-size: 18px; display: flex; align-items: center; gap: 10px;">
            </h3>
            <div style="background: #f8fafc; padding: 20px; border-radius: 12px; line-height: 1.6; color: #1f2937;">
                <div style="font-weight: bold; font-size: 16px; margin-bottom: 10px; color: { 'red' if color == 'high' else 'orange' if color == 'medium' else 'green'};">
                    {emoji} <strong>ANÁLISIS DE RIESGO - {risk_level}</strong>
                </div>
                <p style="margin-bottom: 15px;">
                    No se identifica riesgo significativo de {condition} (probabilidad: <strong>{probability:.1%}</strong>).
                </p>
                <p style="margin-bottom: 15px; color: #4b5563;">
                    {adjustment_info}
                </p>
        """
        
        if clinical_factors:
            analysis_html += f"""
                <div style="margin-top: 15px;">
                    <div style="font-weight: bold; margin-bottom: 8px; color: #374151;">Factores considerados:</div>
                    <ul style="margin: 0; padding-left: 20px; color: #4b5563;">
                        {''.join([f'<li style="margin-bottom: 5px;">{factor}</li>' for factor in clinical_factors])}
                    </ul>
                </div>
                <p style="margin-top: 10px; color: #4b5563;">
                    Aunque se identificaron algunos factores, el análisis general no indica riesgo significativo.
                </p>
            """
        
        analysis_html += f"""
                <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                    <div style="font-weight: bold; color: #1e40af;">📊 Recomendación:</div>
                    <div style="color: #374151; margin-top: 5px;">{risk_desc}</div>
                </div>
            </div>
        </div>
        """
    
    # IMPORTANTE: Incluir tanto 'analysis' como 'analysis_html' para compatibilidad
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': risk_level,
        'risk_description': risk_desc,
        'risk_color': color,
        'color_gradient': color_gradient,
        'analysis': analysis_html,  # Para compatibilidad con código existente
        'analysis_html': analysis_html,  # Nueva clave con HTML formateado
        'factors': clinical_factors,
        'was_adjusted': bool(was_adjusted),
        'base_probability': float(base_probability),
        'risk_multiplier': float(risk_multiplier),
        'model_name': model_name,
        'condition': condition,
        'emoji': emoji
    }

# ============================
# RUTAS API
# ============================

@app.route('/api/predict/stroke', methods=['POST', 'OPTIONS'])
def predict_stroke():
    try:
        print("\n" + "=" * 40)
        print("🧠 PREDICCIÓN STROKE")
        print("=" * 40)
        
        # CORS preflight
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
            
        print(f"📥 Datos recibidos: {data}")
        
        if stroke_model is None:
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        # Preprocesar
        df_input = preprocess_stroke_data(data)
        print(f"📊 Datos preprocesados")
        
        # Predicción original del modelo
        original_prediction = int(stroke_model.predict(df_input)[0])
        original_probability = float(stroke_model.predict_proba(df_input)[0][1])
        
        print(f"🤖 Modelo original: Pred={original_prediction}, Prob={original_probability:.2%}")
        
        # Aplicar corrección clínica
        adjustment = ClinicalRiskAdjuster.adjust_stroke_risk(
            data, original_prediction, original_probability
        )
        
        print(f"⚖️  Ajuste clínico: Multiplicador={adjustment['risk_multiplier']}x")
        print(f"📈 Probabilidad ajustada: {adjustment['probability']:.2%}")
        
        # Generar análisis
        result = generate_analysis(adjustment, 'stroke', data, original_prediction)
        
        response = {
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'risk_description': result['risk_description'],
            'risk_color': result['risk_color'],
            'analysis': result['analysis'],
            'factors': result['factors'],
            'debug_info': {
                'original_prediction': original_prediction,
                'original_probability': original_probability,
                'was_adjusted': result['was_adjusted'],
                'risk_multiplier': result['risk_multiplier'],
                'clinical_factors_count': len(result['factors'])
            }
        }
        
        print(f"✅ Respuesta generada")
        print("=" * 40)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/heart', methods=['POST', 'OPTIONS'])
def predict_heart():
    try:
        print("\n" + "=" * 40)
        print("❤️ PREDICCIÓN HEART")
        print("=" * 40)
        
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
            
        print(f"📥 Datos recibidos: {data}")
        
        if heart_model is None:
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        # Preprocesar
        df_input = preprocess_heart_data(data)
        print(f"📊 Datos preprocesados")
        
        # Predicción original
        original_prediction = int(heart_model.predict(df_input)[0])
        original_probability = float(heart_model.predict_proba(df_input)[0][1])
        
        print(f"🤖 Modelo original: Pred={original_prediction}, Prob={original_probability:.2%}")
        
        # Aplicar corrección clínica
        adjustment = ClinicalRiskAdjuster.adjust_heart_risk(
            data, original_prediction, original_probability
        )
        
        print(f"⚖️  Ajuste clínico: Multiplicador={adjustment['risk_multiplier']}x")
        print(f"📈 Probabilidad ajustada: {adjustment['probability']:.2%}")
        
        # Generar análisis
        result = generate_analysis(adjustment, 'heart', data, original_prediction)
        
        response = {
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'risk_description': result['risk_description'],
            'risk_color': result['risk_color'],
            'analysis': result['analysis'],
            'factors': result['factors'],
            'debug_info': {
                'original_prediction': original_prediction,
                'original_probability': original_probability,
                'was_adjusted': result['was_adjusted'],
                'risk_multiplier': result['risk_multiplier'],
                'clinical_factors_count': len(result['factors'])
            }
        }
        
        print(f"✅ Respuesta generada")
        print("=" * 40)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'stroke': stroke_model is not None,
            'heart': heart_model is not None
        },
        'features': {
            'clinical_adjustment': True,
            'cors_enabled': True
        }
    })

@app.route('/api/test/stroke', methods=['GET'])
def test_stroke_endpoint():
    """Prueba con caso de alto riesgo"""
    test_data = {
        'gender': 'Female',
        'age': 67,
        'hypertension': 1,
        'heart_disease': 0,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 'formerly smoked'
    }
    
    try:
        # Simular procesamiento
        df_input = preprocess_stroke_data(test_data)
        
        if stroke_model:
            original_prediction = int(stroke_model.predict(df_input)[0])
            original_probability = float(stroke_model.predict_proba(df_input)[0][1])
            
            adjustment = ClinicalRiskAdjuster.adjust_stroke_risk(
                test_data, original_prediction, original_probability
            )
            
            result = generate_analysis(adjustment, 'stroke', test_data, original_prediction)
            
            return jsonify({
                'test_case': 'Stroke - Alto riesgo',
                'input_data': test_data,
                'result': result
            })
        else:
            return jsonify({'error': 'Modelo no cargado'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test/heart', methods=['GET'])
def test_heart_endpoint():
    """Prueba con caso de alto riesgo"""
    test_data = {
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
    }
    
    try:
        df_input = preprocess_heart_data(test_data)
        
        if heart_model:
            original_prediction = int(heart_model.predict(df_input)[0])
            original_probability = float(heart_model.predict_proba(df_input)[0][1])
            
            adjustment = ClinicalRiskAdjuster.adjust_heart_risk(
                test_data, original_prediction, original_probability
            )
            
            result = generate_analysis(adjustment, 'heart', test_data, original_prediction)
            
            return jsonify({
                'test_case': 'Heart - Caso prueba',
                'input_data': test_data,
                'result': result
            })
        else:
            return jsonify({'error': 'Modelo no cargado'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Página de inicio"""
    return

# ============================
# ============================
# ANÁLISIS DE IMÁGENES (CÁNCER DE PIEL)
# ============================

# Importar bibliotecas necesarias para procesamiento de imágenes
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    import numpy as np
    from PIL import Image
    import io
    print("✅ TensorFlow y bibliotecas de imagen importadas")
except ImportError as e:
    print(f"⚠️ Advertencia: {e}")
    print("   Instala con: pip install tensorflow pillow")

# Rutas para el modelo de imágenes
IMAGES_MODEL_PATH = os.path.join(BACKEND_PATH, "modelo_imagenes.h5")
IMAGES_CLASSES_PATH = os.path.join(BACKEND_PATH, "clases.json")

# Variables globales para el modelo de imágenes
images_model = None
class_names = []

def load_images_model():
    """Cargar modelo de imágenes de cáncer de piel"""
    global images_model, class_names
    
    try:
        if os.path.exists(IMAGES_MODEL_PATH) and os.path.exists(IMAGES_CLASSES_PATH):
            # Cargar modelo
            print(f"📸 Cargando modelo de imágenes desde: {IMAGES_MODEL_PATH}")
            images_model = load_model(IMAGES_MODEL_PATH)
            print("✅ Modelo de imágenes cargado correctamente")
            
            # Cargar nombres de clases
            with open(IMAGES_CLASSES_PATH, 'r') as f:
                class_names = json.load(f)
            print(f"✅ {len(class_names)} clases cargadas: {class_names}")
            
            return True
        else:
            print(f"❌ No se encontraron archivos del modelo de imágenes:")
            print(f"   Modelo: {IMAGES_MODEL_PATH} {'EXISTE' if os.path.exists(IMAGES_MODEL_PATH) else 'NO EXISTE'}")
            print(f"   Clases: {IMAGES_CLASSES_PATH} {'EXISTE' if os.path.exists(IMAGES_CLASSES_PATH) else 'NO EXISTE'}")
            return False
            
    except Exception as e:
        print(f"❌ Error cargando modelo de imágenes: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_image(img_data):
    """Preprocesar imagen para el modelo"""
    try:
        # Abrir imagen desde bytes
        img = Image.open(io.BytesIO(img_data))
        
        # Redimensionar a 224x224 (tamaño MobileNetV2)
        img = img.resize((224, 224))
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convertir a array y preprocesar
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        raise

def predict_skin_cancer(image_data):
    """Realizar predicción de cáncer de piel"""
    global images_model, class_names
    
    try:
        # Preprocesar imagen
        processed_img = preprocess_image(image_data)
        
        # Realizar predicción
        predictions = images_model.predict(processed_img, verbose=0)
        
        # Obtener clase y confianza
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Obtener nombre de la clase
        if class_names and predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Clase {predicted_class_idx}"
        
        # Obtener todas las probabilidades
        all_probabilities = {}
        for i, prob in enumerate(predictions[0]):
            class_name = class_names[i] if i < len(class_names) else f"Clase {i}"
            all_probabilities[class_name] = float(prob)
        
        # Ordenar por probabilidad descendente
        sorted_probabilities = dict(sorted(all_probabilities.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True))
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'class_index': int(predicted_class_idx),
            'all_probabilities': sorted_probabilities
        }
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        raise

def generate_skin_analysis(prediction_result):
    """Generar análisis clínico de la predicción de cáncer de piel - VERSIÓN MEJORADA"""
    
    predicted_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence']
    all_probabilities = prediction_result['all_probabilities']
    
    # IMPORTANTE: Definir claramente qué clases son ALTO riesgo
    high_risk_classes = ['melanoma', 'melanocytic', 'malignant']
    moderate_risk_classes = ['basal_cell_carcinoma', 'squamous_cell_carcinoma', 
                            'actinic_keratosis', 'suspicious']
    low_risk_classes = ['nevus', 'seborrheic_keratosis', 'benign_keratosis',
                       'dermatofibroma', 'vascular_lesion', 'benign']
    
    # Normalizar nombre de clase para búsqueda (minúsculas)
    class_lower = predicted_class.lower()
    
    # Determinar riesgo basado en la CLASE PREDICHA y CONFIANZA
    risk_level = 'MODERADO'  # Default
    
    # Verificar si es ALTO riesgo
    for high_class in high_risk_classes:
        if high_class in class_lower:
            # Si la confianza es alta (>70%), es ALTO riesgo
            if confidence > 0.7:
                risk_level = 'ALTO'
            else:
                risk_level = 'MODERADO-ALTO'
            break
    
    # Si no es alto riesgo, verificar MODERADO
    if risk_level == 'MODERADO':
        for moderate_class in moderate_risk_classes:
            if moderate_class in class_lower:
                # Si la confianza es alta, podría ser moderado-alto
                if confidence > 0.8:
                    risk_level = 'MODERADO-ALTO'
                else:
                    risk_level = 'MODERADO'
                break
    
    # Si no es alto ni moderado, es BAJO
    if risk_level == 'MODERADO':
        for low_class in low_risk_classes:
            if low_class in class_lower:
                risk_level = 'BAJO'
                break
    
    # ADICIONAL: Considerar la confianza para ajustar riesgo
    if confidence < 0.5:
        # Baja confianza, bajar nivel de riesgo
        if risk_level == 'ALTO':
            risk_level = 'MODERADO-ALTO'
        elif risk_level == 'MODERADO-ALTO':
            risk_level = 'MODERADO'
    
    # Ajustar colores y recomendaciones según riesgo
    if 'ALTO' in risk_level:
        color = '#ef4444'
        emoji = '🔴'
        urgency = 'URGENTE'
        recommendation = 'Se recomienda VISITA INMEDIATA al dermatólogo (en menos de 48 horas).'
        explanation = f'Se identificó {predicted_class} con alta probabilidad ({confidence*100:.1f}%). Las características son altamente sospechosas de malignidad y requieren evaluación profesional inmediata.'
    elif 'MODERADO' in risk_level:
        color = '#f59e0b'
        emoji = '🟡'
        urgency = 'PRIORITARIA'
        recommendation = 'Se recomienda consulta dermatológica en las próximas 2 semanas.'
        explanation = f'Se identificó {predicted_class}. Aunque no es claramente maligno, requiere evaluación profesional para descartar riesgos.'
    else:
        color = '#10b981'
        emoji = '🟢'
        urgency = 'ROUTINARIA'
        recommendation = 'Se recomienda monitoreo anual y consulta si hay cambios.'
        explanation = f'Se identificó {predicted_class} de tipo benigno. Mantener seguimiento regular.'
    
    # Mostrar las 3 principales probabilidades
    top_classes = list(all_probabilities.items())[:3]
    
    analysis_html = f"""
    <div style="background:white;border-radius:12px;padding:20px;border:2px solid {color}">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:15px">
            <div style="font-size:32px">{emoji}</div>
            <div style="background:{color};color:white;padding:6px 14px;border-radius:8px;font-weight:700">
                {risk_level} RIESGO
            </div>
            <div style="font-size:14px;color:#666;background:#f8fafc;padding:4px 12px;border-radius:6px;">
                {predicted_class}
            </div>
        </div>
        
        <div style="margin-top:10px;font-size:15px;color:var(--text-primary)">
            <strong>📊 Confianza del modelo:</strong> {(confidence * 100):.1f}%
            <div style="display:inline-block;width:100px;height:6px;background:#e5e7eb;border-radius:3px;margin-left:10px;vertical-align:middle;overflow:hidden;">
                <div style="height:100%;width:{confidence * 100}%;background:{color};border-radius:3px;"></div>
            </div>
        </div>
        
        <div style="margin-top:15px;">
            <strong>🔍 Diagnósticos posibles:</strong>
            <div style="margin-top:8px;background:#f8fafc;padding:12px;border-radius:8px;font-size:14px;">
                {''.join([f'<div style="margin-bottom:5px;display:flex;justify-content:space-between;"><span>{cls}:</span> <span style="font-weight:bold">{(prob*100):.1f}%</span></div>' for cls, prob in top_classes])}
            </div>
        </div>
        
        <div style="margin-top:20px;padding:15px;background:{'#fee2e2' if 'ALTO' in risk_level else '#fef3c7' if 'MODERADO' in risk_level else '#f0fdf4'};border-radius:10px;border-left:4px solid {color};">
            <div style="font-weight:700;color:{'#dc2626' if 'ALTO' in risk_level else '#92400e' if 'MODERADO' in risk_level else '#047857'};margin-bottom:8px;display:flex;align-items:center;gap:8px;">
                <span>{"🚨" if 'ALTO' in risk_level else "⚠️" if 'MODERADO' in risk_level else "✅"}</span> {urgency} - {predicted_class.upper()}
            </div>
            <div style="color:{'#7f1d1d' if 'ALTO' in risk_level else '#92400e' if 'MODERADO' in risk_level else '#065f46'};font-size:14px;line-height:1.5;">
                <strong>{recommendation}</strong><br>
                {explanation}
            </div>
        </div>
        
        <div style="margin-top:15px;padding:10px;background:#f8fafc;border-radius:8px;font-size:13px;color:#666">
            <strong>💡 Guía de riesgo:</strong><br>
            <span style="color:#ef4444">🔴 ALTO:</span> Lesiones sospechosas de cáncer (melanoma, carcinoma)<br>
            <span style="color:#f59e0b">🟡 MODERADO:</span> Lesiones atípicas que requieren evaluación<br>
            <span style="color:#10b981">🟢 BAJO:</span> Lesiones benignas con seguimiento rutinario
        </div>
    </div>
    """
    
    return {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'risk_level': risk_level,
        'urgency': urgency,
        'recommendation': recommendation,
        'analysis_html': analysis_html,
        'all_probabilities': all_probabilities,
        'color': color,
        'emoji': emoji
    }

@app.route('/api/skin/debug', methods=['POST'])
def skin_debug():
    """Endpoint de diagnóstico para ver qué predice realmente el modelo"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se recibió imagen'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        print(f"\n🔍 DEBUG - Analizando imagen: {image_file.filename}")
        
        # Cargar modelo si no está cargado
        if images_model is None:
            if not load_images_model():
                return jsonify({'error': 'Modelo no disponible'}), 500
        
        # Leer imagen
        image_data = image_file.read()
        
        # Preprocesar
        processed_img = preprocess_image(image_data)
        
        # Realizar predicción
        predictions = images_model.predict(processed_img, verbose=0)
        
        # Mostrar TODAS las predicciones
        print("\n📊 TODAS LAS PREDICCIONES:")
        print("-" * 50)
        
        all_results = []
        for i, prob in enumerate(predictions[0]):
            class_name = class_names[i] if i < len(class_names) else f"Clase_{i}"
            all_results.append({
                'class': class_name,
                'probability': float(prob),
                'percentage': float(prob) * 100
            })
            print(f"{class_name:30} {prob*100:6.2f}%")
        
        # Ordenar por probabilidad
        all_results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Predicción principal
        top_result = all_results[0]
        
        print(f"\n🎯 PREDICCIÓN PRINCIPAL:")
        print(f"Clase: {top_result['class']}")
        print(f"Probabilidad: {top_result['probability']:.4f} ({top_result['percentage']:.1f}%)")
        print(f"\n📋 Clases disponibles: {class_names}")
        
        # Generar análisis temporal para ver qué pasa
        temp_prediction = {
            'predicted_class': top_result['class'],
            'confidence': top_result['probability'],
            'all_probabilities': {item['class']: item['probability'] for item in all_results}
        }
        
        analysis = generate_skin_analysis(temp_prediction)
        
        return jsonify({
            'debug': True,
            'all_predictions': all_results,
            'top_prediction': top_result,
            'class_names': class_names,
            'generated_risk_level': analysis['risk_level'],
            'why': f"Clase '{top_result['class']}' con {top_result['percentage']:.1f}% de confianza"
        })
        
    except Exception as e:
        print(f"❌ Error en debug: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/predict/skin', methods=['POST', 'OPTIONS'])
def predict_skin():
    """Endpoint para análisis de cáncer de piel desde imágenes"""
    try:
        print("\n" + "=" * 40)
        print("📸 PREDICCIÓN CÁNCER DE PIEL")
        print("=" * 40)
        
        # CORS preflight
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200
        
        # Verificar si hay archivo
        if 'image' not in request.files:
            return jsonify({'error': 'No se recibió imagen'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        # Verificar formato
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in image_file.filename or \
           image_file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'Formato de imagen no soportado'}), 400
        
        print(f"📥 Imagen recibida: {image_file.filename}")
        
        # Cargar modelo si no está cargado
        if images_model is None:
            print("🔄 Cargando modelo de imágenes...")
            if not load_images_model():
                return jsonify({'error': 'Modelo de imágenes no disponible'}), 500
        
        # Leer imagen
        image_data = image_file.read()
        
        # Realizar predicción
        prediction_result = predict_skin_cancer(image_data)
        
        # Generar análisis clínico
        analysis_result = generate_skin_analysis(prediction_result)
        
        # Construir respuesta
        response = {
            'success': True,
            'prediction': {
                'class': analysis_result['predicted_class'],
                'confidence': analysis_result['confidence'],
                'risk_level': analysis_result['risk_level'],
                'urgency': analysis_result['urgency']
            },
            'analysis_html': analysis_result['analysis_html'],
            'recommendation': analysis_result['recommendation'],
            'probabilities': analysis_result['all_probabilities'],
            'visual': {
                'color': analysis_result['color'],
                'emoji': analysis_result['emoji']
            }
        }
        
        print(f"✅ Predicción completada: {analysis_result['predicted_class']} ({analysis_result['risk_level']})")
        print("=" * 40)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/skin/status', methods=['GET'])
def skin_model_status():
    """Verificar estado del modelo de piel"""
    return jsonify({
        'model_loaded': images_model is not None,
        'classes_loaded': len(class_names) > 0,
        'available_classes': class_names,
        'model_path': IMAGES_MODEL_PATH,
        'classes_path': IMAGES_CLASSES_PATH
    })

# Cargar modelo de imágenes al inicio
print("\n" + "=" * 40)
print("📸 MODELO DE IMÁGENES (CÁNCER DE PIEL)")
print("=" * 40)
load_images_model()


# ============================
# CHAT IA AVANZADO 
# ============================


from openai import OpenAI
import json
from datetime import datetime

# Cargar configuración
try:
    from config import API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    print(f"🔑 Chat API: {'CONECTADO' if API_KEY else 'MODO DEMO'}")
except ImportError:
    print("⚠️ Config.py no encontrado, usando modo demo")
    API_KEY = None
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_BASE_URL = "https://api.openai.com/v1"

# Inicializar cliente OpenAI si hay API Key
openai_client = None
if API_KEY:
    try:
        openai_client = OpenAI(
            api_key=API_KEY,
            base_url=OPENAI_BASE_URL
        )
        print("✅ Cliente OpenAI inicializado")
    except Exception as e:
        print(f"❌ Error inicializando OpenAI: {e}")
        openai_client = None

# Sistema de historial de chat (en memoria, para producción usar DB)
chat_histories = {}

# Prompt de sistema para el asistente médico (MODIFICADO)
MEDICAL_SYSTEM_PROMPT = """Eres Aila Assistant, un asistente médico IA especializado en salud preventiva. 

TU ROL:
1. Proveer información médica general y educativa
2. Ayudar a interpretar términos médicos
3. Ofrecer consejos de estilo de vida saludable
4. Explicar condiciones médicas comunes
5. Guiar sobre cuándo buscar atención médica

REGLAS IMPORTANTES:
- NUNCA dar diagnósticos médicos
- NUNCA prescribir tratamientos
- SIEMPRE recomendar consultar con profesionales médicos
- Ser empático, claro y preciso
- Usar lenguaje médico accesible
- Priorizar seguridad del paciente

FORMATO DE RESPUESTA:
- Usa emojis relevantes 🩺💊🏥 para categorizar la respuesta
- Estructura en secciones claras pero SIN usar **asteriscos** ni markdown
- Usa títulos con emojis seguidos de dos puntos
- Destaca información importante con viñetas
- Usa saltos de línea para separar ideas
- El tono debe ser profesional pero cálido
- Evita lenguaje demasiado técnico sin explicar

EJEMPLOS DE FORMATO:
Usuario: "¿Qué es la hipertensión?"
Tú: "🩺 Hipertensión Arterial

La hipertensión es cuando la presión en tus arterias está persistentemente elevada...

💡 Recomendaciones:
• Controla tu presión regularmente
• Reduce el consumo de sal
• Mantén un peso saludable

🏥 Consulta médica: Si tienes síntomas o dudas, consulta con un cardiólogo para evaluación personalizada."

Usuario: "Me duele la cabeza"
Tú: "⚠️ Dolor de Cabeza

Los dolores de cabeza pueden tener muchas causas...

🚨 Busca atención médica inmediata si:
• Dolor súbito e intenso
• Fiebre alta
• Confusión o pérdida de conciencia

💊 Cuidados generales:
• Descanso adecuado
• Buena hidratación
• Ambiente tranquilo

🏥 Importante: Consulta a tu médico para evaluación específica."

Ahora responde la siguiente consulta:"""

def get_chat_response(user_message, user_id="default", conversation_id=None):
    """Obtener respuesta de OpenAI para consulta médica"""
    
    # Si no hay API Key o cliente, usar respuestas predefinidas
    if not openai_client:
        return get_fallback_response(user_message)
    
    try:
        # Construir historial de conversación
        messages = [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT}
        ]
        
        # Agregar historial previo si existe
        if conversation_id and conversation_id in chat_histories:
            for msg in chat_histories[conversation_id][-5:]:  # Últimos 5 mensajes
                messages.append(msg)
        
        # Agregar mensaje actual del usuario
        messages.append({"role": "user", "content": user_message})
        
        print(f"🤖 Enviando consulta a OpenAI: {user_message[:50]}...")
        
        # Llamar a OpenAI API
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )
        
        # Obtener respuesta
        ai_response = response.choices[0].message.content
        
        # Guardar en historial
        if conversation_id:
            if conversation_id not in chat_histories:
                chat_histories[conversation_id] = []
            
            chat_histories[conversation_id].append({"role": "user", "content": user_message})
            chat_histories[conversation_id].append({"role": "assistant", "content": ai_response})
            
            # Limitar historial a 20 mensajes
            if len(chat_histories[conversation_id]) > 20:
                chat_histories[conversation_id] = chat_histories[conversation_id][-20:]
        
        print(f"✅ Respuesta recibida: {ai_response[:50]}...")
        return ai_response
        
    except Exception as e:
        print(f"❌ Error con OpenAI API: {e}")
        return get_fallback_response(user_message)

def get_fallback_response(user_message):
    """Respuestas predefinidas cuando OpenAI no está disponible (MODIFICADO)"""
    
    fallback_responses = [
        "🩺 Información General\n\nEntiendo tu consulta sobre salud. Para brindarte la mejor orientación:\n\n📋 Detalles importantes:\n• ¿Podrías darme más detalles sobre tu situación?\n• ¿Qué síntomas específicos estás experimentando?\n• ¿Desde cuándo tienes estos síntomas?\n\n💡 Recuerda: Esta información es educativa. Para diagnóstico y tratamiento, consulta con un profesional médico.\n\n🏥 Tu salud es nuestra prioridad.",
        
        "🏥 Asistencia Médica\n\nTu salud es importante. Basado en tu pregunta:\n\n🔍 Recomendaciones generales:\n• Mantén un registro detallado de síntomas\n• Programa consulta con tu médico de cabecera\n• Realiza chequeos preventivos regulares\n\n⚠️ Señales de alerta:\n• Dolor intenso o repentino\n• Fiebre alta persistente\n• Dificultad para respirar\n• Sangrado inexplicable\n\n🚨 Si experimentas alguno de estos síntomas, busca atención médica inmediata",
        
        "💊 Consejos de Salud Preventiva\n\nPara mantener una buena salud:\n\n✅ Hábitos saludables esenciales:\n1. Dieta balanceada rica en frutas y verduras\n2. Ejercicio regular (30 minutos diarios)\n3. Sueño adecuado (7-9 horas)\n4. Manejo efectivo del estrés\n\n🩺 Prevención proactiva:\n• Chequeos médicos anuales\n• Vacunación al día\n• Autoexámenes regulares\n\n📋 Para tu consulta específica, te sugiero:\n• Anotar todas tus dudas\n• Llevar esta información a tu médico\n• Solicitar evaluación personalizada"
    ]
    
    import random
    return random.choice(fallback_responses)

def generate_conversation_html(conversation_id=None):
    """Generar HTML para mostrar historial de conversación"""
    
    if not conversation_id or conversation_id not in chat_histories:
        return ""
    
    html_parts = []
    for msg in chat_histories[conversation_id][-10:]:  # Últimos 10 mensajes
        if msg["role"] == "user":
            html_parts.append(f"""
                <div class="msg user">
                    <div class="msg-content">{msg["content"]}</div>
                    <div class="msg-time">{datetime.now().strftime("%H:%M")}</div>
                </div>
            """)
        else:
            html_parts.append(f"""
                <div class="msg bot">
                    <div class="msg-header">
                        <strong>Asistente IA:</strong>
                    </div>
                    <div class="msg-content">{msg["content"]}</div>
                    <div class="msg-time">{datetime.now().strftime("%H:%M")}</div>
                </div>
            """)
    
    return "".join(html_parts)

@app.route('/api/chat/send', methods=['POST', 'OPTIONS'])
def chat_send():
    """Endpoint para enviar mensaje al chat"""
    try:
        print("\n" + "=" * 40)
        print("💬 CHAT - Nuevo mensaje")
        print("=" * 40)
        
        # CORS preflight
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
        
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Mensaje vacío'}), 400
        
        print(f"📤 Usuario: {user_message}")
        
        # Obtener respuesta de IA
        ai_response = get_chat_response(user_message, conversation_id=conversation_id)
        
        # Guardar timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        response_data = {
            'success': True,
            'response': ai_response,
            'conversation_id': conversation_id,
            'timestamp': timestamp,
            'is_openai': openai_client is not None,
            'message_count': len(chat_histories.get(conversation_id, [])) if conversation_id in chat_histories else 0
        }
        
        print(f"📥 Respuesta generada ({'OpenAI' if openai_client else 'Demo'})")
        print("=" * 40)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error en chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
def chat_history():
    """Obtener historial de chat"""
    conversation_id = request.args.get('conversation_id', 'default')
    
    if conversation_id in chat_histories:
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'messages': chat_histories[conversation_id][-20:],  # Últimos 20 mensajes
            'total_messages': len(chat_histories[conversation_id])
        })
    else:
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'messages': [],
            'total_messages': 0
        })

@app.route('/api/chat/new', methods=['POST'])
def new_chat():
    """Iniciar nueva conversación"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        
        # Crear nuevo ID de conversación
        new_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Inicializar historial vacío
        chat_histories[new_id] = []
        
        return jsonify({
            'success': True,
            'conversation_id': new_id,
            'timestamp': datetime.now().isoformat(),
            'message': 'Nueva conversación iniciada'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/status', methods=['GET'])
def chat_status():
    """Verificar estado del chat"""
    return jsonify({
        'status': 'online',
        'openai_available': openai_client is not None,
        'model': OPENAI_MODEL if openai_client else 'demo',
        'active_conversations': len(chat_histories),
        'total_messages': sum(len(msgs) for msgs in chat_histories.values())
    })

# ============================
# ACTUALIZAR INICIO DEL SERVIDOR
# ============================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("✅ SERVIDOR CONFIGURADO")
    print("=" * 60)
    print("🌐 URL: http://127.0.0.1:5000")
    print("📋 Endpoints principales:")
    print("  POST /api/predict/stroke   - Derrame cerebral")
    print("  POST /api/predict/heart    - Enfermedad cardíaca")
    print("  POST /api/predict/skin     - Cáncer de piel (imágenes)")
    print("  GET  /api/health           - Verificación")
    print("  GET  /api/skin/status      - Estado modelo imágenes")
    print("=" * 60)
    print("📸 Sistema incluye modelo real de cáncer de piel")
    print("⚖️  Con ajuste clínico para datos desbalanceados")
    print("=" * 60)
    
    app.run(debug=True, port=5000, host='127.0.0.1')