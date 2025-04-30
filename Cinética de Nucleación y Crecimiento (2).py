# Cinética de Nucleación y Crecimiento
from scipy.constants import Boltzmann as kB
import matplotlib.pyplot as plt
import numpy as np

# Parámetros para la solubilidad (Van't Hoff)
ΔH_sol = 45.6e3   # J/mol (entalpía de solución)
R = 8.314         # J/(mol·K)
C0 = 5.2          # mg/mL (solubilidad a T0)
T0 = 298.15       # K (25°C)

# Rango de temperaturas para la solubilidad
T_sol = np.linspace(293, 313, 50)  # 20°C a 40°C

# Ecuación de Van't Hoff para calcular Cs
Cs = C0 * np.exp(-(ΔH_sol/R) * (1/T_sol - 1/T0))

# Parámetros para nucleación y crecimiento
kn = 1e8        # Constante de nucleación (1/s)
kg = 0.05       # Constante de crecimiento (µm/min)
ΔG = 1.2e-19    # J (energía libre crítica)
g = 1.5         # Orden de crecimiento

# Rango de sobresaturación (S = C/Cs)
S = np.linspace(1.1, 2.0, 50)

# Temperatura fija para este ejemplo (30°C = 303.15 K)
T = 303.15  # K

# Calcular la concentración de saturación promedio a la temperatura de interés
indice_T_objetivo = np.argmin(np.abs(T_sol - T))
Cs_a_T = Cs[indice_T_objetivo]

C = S * Cs_a_T

# Tasa de nucleación (B) - ¡Corregido para tener la misma dimensión que S!
B_valor_unico = kn * np.exp(-ΔG / (kB * T))
B = np.full_like(S, B_valor_unico) # Crea un array lleno con el valor de B

# Tasa de crecimiento (G)
G = kg * (C - Cs_a_T)**g

# Gráficas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(S, B, 'r-', linewidth=2)
ax1.set_xlabel('Sobresaturación (S)')
ax1.set_ylabel('Tasa de Nucleación (B)')
ax1.set_title('Nucleación vs Sobresaturación')
ax1.grid(True)

ax2.plot(S, G, 'g-', linewidth=2)
ax2.set_xlabel('Sobresaturación (S)')
ax2.set_ylabel('Tasa de Crecimiento (G)')
ax2.set_title('Crecimiento vs Sobresaturación')
ax2.grid(True)

plt.tight_layout()
plt.show()