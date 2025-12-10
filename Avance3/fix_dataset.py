import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('student_sleep_patterns_extended_with_performance.csv')

# Ensure Gender_Encoded is correct (Men=1, Women=0)
# 'Male' is usually 1. 
df['Gender_Encoded'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# New Jurca Formula provided by User
# MET=[sexo x (2,77)-edad x (0,10)-IMC x (0,17)-FCr x(0,03)+CAF x (1,0)]+18,07
df['CRF_METs'] = (
    (df['Gender_Encoded'] * 2.77) - 
    (df['Age'] * 0.10) - 
    (df['BMI'] * 0.17) - 
    (df['Resting_Heart_Rate'] * 0.03) + 
    (df['Activity_Score'] * 1.0) + 
    18.07
)

# Recalculate Physical Capacity Score (0-100)
# Updated Scale for new formula: Min=10 METs (Low), Max=24 METs (Elite) based on dataset stats (Mean~17)
df['Score_Physical_Capacity'] = ((df['CRF_METs'] - 10) / (24 - 10)) * 100
df['Score_Physical_Capacity'] = df['Score_Physical_Capacity'].clip(0, 100)

# Recalculate Habits Score (Same as Codigo1ia.py)
df['Score_Sleep_Qty'] = ((df['Sleep_Duration'] - 4) / 4) * 100
df['Score_Sleep_Qty'] = df['Score_Sleep_Qty'].clip(0, 100)

df['Score_Sleep_Qual'] = df['Sleep_Quality'] * 10 
df['Score_Sleep_Qual'] = df['Score_Sleep_Qual'].clip(0, 100)

df['Score_Nutrition'] = 100 - (abs(df['Caloric_Intake'] - 2500) / 2500 * 100)
df['Score_Nutrition'] = df['Score_Nutrition'].clip(0, 100)

df['Score_Water'] = (df['Water_Liters'] / 3.0) * 100
df['Score_Water'] = df['Score_Water'].clip(0, 100)

df['Score_Habits'] = (
    (df['Score_Sleep_Qty'] * 0.3) + 
    (df['Score_Sleep_Qual'] * 0.2) + 
    (df['Score_Nutrition'] * 0.3) + 
    (df['Score_Water'] * 0.2)
)
df['Score_Habits'] = df['Score_Habits'].clip(0, 100)

# Final Performance Score (60% Physical, 40% Habits)
df['SportsPerformanceScore'] = (df['Score_Physical_Capacity'] * 0.60) + (df['Score_Habits'] * 0.40)
df['SportsPerformanceScore'] = df['SportsPerformanceScore'].clip(0, 100)

# Save
df.to_csv('student_sleep_patterns_extended_with_performance.csv', index=False)
print("Dataset updated successfully with new Jurca formula.")
print(df[['Gender', 'Age', 'BMI', 'CRF_METs', 'SportsPerformanceScore']].head())
