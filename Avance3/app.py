from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar Modelo
MODEL_PATH = 'modelo_mlp.pkl'
SCALER_PATH = 'scaler.pkl'
modelo = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    mensaje = ""
    tipo_alerta = ""
    jurca_feedback = ""
    
    # Valores por defecto
    datos = {
        'age': 22, 'gender': 1, 'weight': 70, 'height': 175, 'resting_hr': 60,
        'sleep': 7.5, 'activity_min': 60, 'calories': 2500, 'water': 2.5, 'quality': 7
    }

    if request.method == 'POST' and modelo:
        try:
            # 1. Obtener Datos
            age = float(request.form['age'])
            gender = int(request.form['gender']) # 1 Male, 0 Female
            weight = float(request.form['weight'])
            height_cm = float(request.form['height'])
            resting_hr = float(request.form['resting_hr'])
            
            sleep = float(request.form['sleep'])
            activity_min = float(request.form['activity_min'])
            calories = float(request.form['calories'])
            water = float(request.form['water'])
            quality = float(request.form['quality'])

            datos = request.form # Guardar para la vista

            # 2. CÁLCULOS INTERMEDIOS (Feature Engineering)
            
            # A) Calcular BMI (IMC)
            height_m = height_cm / 100
            bmi = weight / (height_m ** 2)

            # B) Convertir Minutos Actividad a "Activity Score" (Escala Jurca 1-7)
            # 0=Sedentario, 7=Atleta Élite
            if activity_min < 30: act_score = 1
            elif activity_min < 60: act_score = 3
            elif activity_min < 90: act_score = 5
            else: act_score = 7

            # 3. PREPARAR VECTOR DE ENTRADA
            # Orden exacto del train_model.py:
            # ['Age', 'Gender_Encoded', 'BMI', 'Resting_Heart_Rate', 'Activity_Score', 
            #  'Sleep_Duration', 'Caloric_Intake', 'Water_Liters', 'Sleep_Quality']
            
            features = np.array([[
                age, gender, bmi, resting_hr, act_score,
                sleep, calories, water, quality
            ]])
            
            # 4. PREDECIR
            features_scaled = scaler.transform(features)
            resultado = modelo.predict(features_scaled)[0]
            prediccion = round(min(100, max(0, resultado)), 1)

            # 5. FEEDBACK
            # Cálculo de Jurca solo para mostrar dato curioso al usuario
            # Cálculo de Jurca actualizado
            jurca_mets = (gender * 2.77) - (age * 0.10) - (bmi * 0.17) - (resting_hr * 0.03) + (act_score * 1.0) + 18.07
            jurca_feedback = f"Tu capacidad estimada (Jurca) es de {jurca_mets:.1f} METs."

            if prediccion >= 85:
                mensaje = "¡Nivel Atleta Profesional! Tu fisiología y hábitos están al máximo."
                tipo_alerta = "success"
            elif prediccion >= 60:
                mensaje = "Buen rendimiento. Tienes buena base física."
                tipo_alerta = "primary"
            else:
                mensaje = "Rendimiento bajo. Mejora tu IMC o tus horas de sueño."
                tipo_alerta = "danger"

        except ValueError:
            mensaje = "Error: Revisa que todos los campos sean números."
            tipo_alerta = "danger"

    return render_template('index.html', prediccion=prediccion, mensaje=mensaje, tipo_alerta=tipo_alerta, datos=datos, jurca_feedback=jurca_feedback)

if __name__ == '__main__':
    app.run(debug=True)