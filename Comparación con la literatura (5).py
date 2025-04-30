import matplotlib.pyplot as plt

# Datos de literatura (ejemplo)
T_lit = [25, 30, 35]  # Chen et al.
size_lit = [60, 70, 80]

# Tus datos
T_sim = [20, 25, 30, 35, 40]
size_sim = [45, 60, 75, 55, 40]

plt.plot(T_sim, size_sim, 'bo-', label='Nuestro Modelo')
plt.plot(T_lit, size_lit, 'r*--', label='Chen et al. (2011)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Tamaño de Cristales (µm)')
plt.legend()
plt.grid(True)
plt.show()
