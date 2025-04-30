# Tamaño de Cristales
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np

# Parámetros de distribución log-normal (ejemplo)
mu = np.log(50)    # Tamaño medio (µm)
sigma = 0.3        # Desviación estándar
samples = 1000     # Número de cristales

# Generar distribución
sizes = lognorm.rvs(s=sigma, scale=np.exp(mu), size=samples)

# Gráfica
plt.figure(figsize=(8, 5))
plt.hist(sizes, bins=30, density=True, alpha=0.6, color='purple')
plt.xlabel('Tamaño de Cristales (µm)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Tamaños de Cristales (PSD)')
plt.grid(True)
plt.show()
