import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. CARGA DE DATOS
print("1. [Pipeline] Cargando dataset...")
try:
    # Usamos el dataset que ya tiene los datos biométricos
    df = pd.read_csv('student_sleep_patterns_extended_with_performance.csv')
    print(f"   Datos cargados: {len(df)} registros.")
except FileNotFoundError:
    print("ERROR: No se encuentra el CSV.")
    exit()

# ---------------------------------------------------------
# 2. IMPLEMENTACIÓN ECUACIÓN DE JURCA (Capacidad Física)
# ---------------------------------------------------------
print("2. [Ciencia] Aplicando Ecuación de Jurca y Recalculando Target...")

# A) Calculamos METs estimados (Capacidad Cardiorespiratoria)
# Fórmula Jurca: METs = 13.08 - (0.074*Age) - (0.057*RHR) + (0.634*ActivityScore) - (0.153*BMI) + (0.589*Gender)
# Nota: Gender en fórmula Jurca suele ser: 1 Hombre, 0 Mujer. Aseguramos que Gender_Encoded cumpla esto.

# En tu CSV ya viene calculado 'CRF_METs', pero lo recalculamos para asegurar la lógica en el código
df['Jurca_METs'] = (
    13.08 - 
    (0.074 * df['Age']) - 
    (0.057 * df['Resting_Heart_Rate']) + 
    (0.634 * df['Activity_Score']) - 
    (0.153 * df['BMI']) + 
    (0.589 * df['Gender_Encoded'])
)

# Normalizamos los METs a una escala 0-100 para mezclarlo con el rendimiento
# Un valor de METs de 15 es atleta de élite (100 pts), un valor de 5 es sedentario (0 pts).
df['Score_Physical_Capacity'] = ((df['Jurca_METs'] - 5) / (15 - 5)) * 100
df['Score_Physical_Capacity'] = df['Score_Physical_Capacity'].clip(0, 100)


# ---------------------------------------------------------
# 3. CÁLCULO DE SCORE DE HÁBITOS (Estilo de Vida - 40%)
# ---------------------------------------------------------

# A. Sueño (Cantidad)
df['Score_Sleep_Qty'] = ((df['Sleep_Duration'] - 4) / 4) * 100
df['Score_Sleep_Qty'] = df['Score_Sleep_Qty'].clip(0, 100)

# B. Sueño (Calidad - Directo del 1 al 10)
df['Score_Sleep_Qual'] = df['Sleep_Quality'] * 10 
df['Score_Sleep_Qual'] = df['Score_Sleep_Qual'].clip(0, 100)

# C. Nutrición (Balance)
df['Score_Nutrition'] = 100 - (abs(df['Caloric_Intake'] - 2500) / 2500 * 100)
df['Score_Nutrition'] = df['Score_Nutrition'].clip(0, 100)

# D. Hidratación (Litros)
df['Score_Water'] = (df['Water_Liters'] / 3.0) * 100
df['Score_Water'] = df['Score_Water'].clip(0, 100)

# PROMEDIO DE HÁBITOS (4 FACTORES)
# Aquí incluimos los 4 factores de estilo de vida equitativamente
df['Score_Habits'] = (
    (df['Score_Sleep_Qty'] * 0.3) + 
    (df['Score_Sleep_Qual'] * 0.2) + 
    (df['Score_Nutrition'] * 0.3) + 
    (df['Score_Water'] * 0.2)
)
df['Score_Habits'] = df['Score_Habits'].clip(0, 100)

# La actividad física NO se suma aquí porque ya es el motor principal
# de la Ecuación de Jurca (Score_Physical_Capacity).


# ---------------------------------------------------------
# 4. VARIABLE OBJETIVO FINAL (Y)
# ---------------------------------------------------------
# El Rendimiento Deportivo Real es una mezcla de tu "Motor" (Jurca) y tu "Gasolina" (Hábitos)
# Le damos peso: 60% a la Capacidad Física (Genética/Entreno) + 40% al Estado Actual (Sueño/Comida)
df['SportsPerformanceFinal'] = (df['Score_Physical_Capacity'] * 0.60) + (df['Score_Habits'] * 0.40)

# Añadimos ruido aleatorio
df['SportsPerformanceFinal'] += np.random.normal(0, 1.5, len(df))
df['SportsPerformanceFinal'] = df['SportsPerformanceFinal'].clip(0, 100)


# ---------------------------------------------------------
# 5. ENTRENAMIENTO
# ---------------------------------------------------------
# AHORA EL MODELO RECIBE MUCHOS MÁS DATOS (BIOMÉTRICOS + HÁBITOS)
features = [
    'Age', 'Gender_Encoded', 'BMI', 'Resting_Heart_Rate', 'Activity_Score', # Factores Jurca
    'Sleep_Duration', 'Caloric_Intake', 'Water_Liters', 'Sleep_Quality'     # Factores Hábitos
]

X = df[features]
y = df['SportsPerformanceFinal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"3. [IA] Entrenando MLP con {len(features)} variables de entrada...")
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=5000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluación
y_pred = mlp.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f"   -> R2 Score: {r2:.4f}")

joblib.dump(mlp, 'modelo_mlp.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("4. [Sistema] Archivos actualizados.")