import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# =============================================
# 1. Definir el perfil de enfriamiento controlado
# =============================================
def cooling_profile(T_initial=40.0, cooling_rate=0.5, t_total=30):
    """
    Simula un perfil lineal de enfriamiento.
    
    Args:
        T_initial (float): Temperatura inicial (°C).
        cooling_rate (float): Tasa de enfriamiento (°C/min).
        t_total (float): Tiempo total de simulación (min).
    
    Returns:
        tuple: (tiempo, temperatura)
    """
    time = np.linspace(0, t_total, 100)  # Vector de tiempo
    temperature = T_initial - cooling_rate * time  # Enfriamiento lineal
    
    # Asegurar que la temperatura no sea negativa
    temperature = np.maximum(temperature, 0)
    
    return time, temperature

# Generar perfil (40°C → 25°C a 0.5°C/min)
time, temperature = cooling_profile(T_initial=40.0, cooling_rate=0.5, t_total=30)

# Gráfica del perfil
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, 'b-', linewidth=2)
plt.xlabel('Tiempo (min)')
plt.ylabel('Temperatura (°C)')
plt.title('Perfil de Enfriamiento Controlado (0.5°C/min)')
plt.grid(True)
plt.show()

# =============================================
# 2. Acoplar el enfriamiento con la cinética de cristalización
# =============================================
def crystallization_kinetics(T, t):
    """
    Modela la cinética de cristalización durante el enfriamiento.
    Incluye nucleación y crecimiento dependientes de la temperatura.
    """
    # Parámetros del modelo (ajustar según datos experimentales)
    kn = 1e8 * np.exp(-5000 / (T + 273.15))  # Efecto Arrhenius en nucleación
    kg = 0.05 * (T / 30.0)**2  # Crecimiento más lento a bajas T
    
    # Concentración de saturación (Van't Hoff simplificado)
    Cs = 5.2 * np.exp(-45.6e3 / (8.314 * (T + 273.15)))  # mg/mL
    
    # Sobresaturación (asumiendo concentración inicial constante)
    C0 = 8.0  # mg/mL
    S = C0 / Cs
    
    # Tasas de nucleación (B) y crecimiento (G)
    B = kn * S**1.5  # 1/(m³·s)
    G = kg * (S - 1)**1.0  # µm/min
    
    return B, G, S

# Simular durante el enfriamiento
B_values = []
G_values = []
S_values = []
for T in temperature:
    B, G, S = crystallization_kinetics(T, t=0)
    B_values.append(B)
    G_values.append(G)
    S_values.append(S)

# Gráficas de resultados (SEPARADAS)
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, 'r-')
plt.xlabel('Tiempo (min)')
plt.ylabel('Temperatura (°C)')
plt.title('Perfil de Temperatura')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time, S_values, 'b-')
plt.xlabel('Tiempo (min)')
plt.ylabel('Sobresaturación (S)')
plt.title('Sobresaturación vs. Tiempo')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time, B_values, 'g-', label='Nucleación (B)')
plt.plot(time, G_values, 'm-', label='Crecimiento (G)')
plt.xlabel('Tiempo (min)')
plt.ylabel('Tasas (unidades normalizadas)')
plt.title('Tasas de Nucleación y Crecimiento vs. Tiempo')
plt.legend()
plt.grid(True)
plt.show()


# =============================================
# 3. Simular la distribución de tamaños durante el enfriamiento
# =============================================
def simulate_crystal_growth(time, temperature, G_values):
    """
    Simula el crecimiento acumulado de cristales durante el enfriamiento.
    """
    crystal_size = np.zeros_like(time)
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        crystal_size[i] = crystal_size[i-1] + G_values[i] * dt
    
    return crystal_size

crystal_size = simulate_crystal_growth(time, temperature, G_values)

plt.figure(figsize=(10, 5))
plt.plot(time, crystal_size, 'k-', linewidth=2)
plt.xlabel('Tiempo (min)')
plt.ylabel('Tamaño de Cristal (µm)')
plt.title('Crecimiento Acumulado de Cristales durante el Enfriamiento')
plt.grid(True)
plt.show()