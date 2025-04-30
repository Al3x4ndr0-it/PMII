# Cálculo de Solubilidad
import numpy as np
import matplotlib.pyplot as plt

# Parámetros (ejemplo para Sulfato de Atazanavir)
ΔH_sol = 45.6e3  # J/mol (entalpía de solución)
R = 8.314       # J/(mol·K)
C0 = 5.2        # mg/mL (solubilidad a T0)
T0 = 298.15     # K (25°C)

# Rango de temperaturas
T = np.linspace(293, 313, 50)  # 20°C a 40°C

# Ecuación de Van't Hoff
Cs = C0 * np.exp(-(ΔH_sol/R) * (1/T - 1/T0))

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(T - 273.15, Cs, 'b-', linewidth=2)  # Convertir K a °C
plt.xlabel('Temperatura (°C)')
plt.ylabel('Solubilidad (mg/mL)')
plt.title('Solubilidad vs Temperatura (Van\'t Hoff)')
plt.grid(True)
plt.show()