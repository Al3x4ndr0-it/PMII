# Optimización
import pandas as pd
import matplotlib.pyplot as plt

# Datos de entrada (ejemplo)
data = {
    'Temperatura (°C)': [20, 25, 30, 35, 40],
    'Sobresaturación (S)': [1.2, 1.3, 1.4, 1.5, 1.6],
    'Tamaño Promedio (µm)': [45, 60, 75, 55, 40],
    'Pureza (%)': [98, 97, 95, 93, 90]
}

df = pd.DataFrame(data)

# Gráfica de optimización
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(df['Temperatura (°C)'], df['Tamaño Promedio (µm)'], 'bo-', label='Tamaño (µm)')
ax2.plot(df['Temperatura (°C)'], df['Pureza (%)'], 'rs--', label='Pureza (%)')

ax1.set_xlabel('Temperatura (°C)')
ax1.set_ylabel('Tamaño Promedio (µm)', color='b')
ax2.set_ylabel('Pureza (%)', color='r')
plt.title('Optimización: Temperatura vs Tamaño y Pureza')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()