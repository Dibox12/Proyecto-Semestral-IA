import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
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
# Fórmula Jurca Corregida: MET = [sexo x (2,77)-edad x (0,10)-IMC x (0,17)-FCr x(0,03)+CAF x (1,0)]+18,07
df['Jurca_METs'] = (
    (df['Gender_Encoded'] * 2.77) - 
    (df['Age'] * 0.10) - 
    (df['BMI'] * 0.17) - 
    (df['Resting_Heart_Rate'] * 0.03) + 
    (df['Activity_Score'] * 1.0) + 
    18.07
)

# Normalizamos los METs a una escala 0-100
# Nueva Escala (Ajustada a distribución de Jurca V2): 10 = 0 pts, 24 = 100 pts.
df['Score_Physical_Capacity'] = ((df['Jurca_METs'] - 10) / (24 - 10)) * 100
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

print(f"3. [IA] Optimizando Hiperparámetros con GridSearch...")
# Definimos el espacio de búsqueda
param_grid = {
    'hidden_layer_sizes': [(128, 64), (128, 128), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01], # Regularización L2
    'learning_rate_init': [0.001, 0.01]
}

mlp = MLPRegressor(solver='adam', max_iter=5000, random_state=42, early_stopping=True)

# Búsqueda (Cross-Validation de 5 pliegues)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"   Mejores Parámetros: {grid_search.best_params_}")

# Usamos el mejor modelo
mlp = best_model

# Evaluación
y_pred = mlp.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f"   -> R2 Score: {r2:.4f}")

print("-" * 50)
print(f"5. [IA] Análisis Final de Métricas (Post-Entrenamiento)...")

# 1. Clustering Final (sobre datos de entrenamiento para ver estructura aprendida/existente)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_scaled)
sil_score = silhouette_score(X_train_scaled, clusters)
print(f"   -> Clustering Final (Silhouette Score): {sil_score:.4f}")

# 2. Métricas del Modelo Final
mse_final = mean_squared_error(y_test, y_pred)
print(f"   -> Error Final (MSE): {mse_final:.4f}")
print(f"   -> Precisión Final (R2): {r2:.4f}")
print(f"   -> Convergencia Final (Loss): {mlp.loss_:.4f}")
print(f"   -> Iteraciones Totales: {mlp.n_iter_}")
print("-" * 50)

joblib.dump(mlp, 'modelo_mlp.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("6. [Sistema] Archivos actualizados.")