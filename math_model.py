import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import json

stages = [
    {'only_stage': 47554, 'fuel': 108_389, 'burn_time': 100, 'f_tract': 2_664_700},
    {'only_stage': 14769, 'fuel': 18633, 'burn_time': 117, 'f_tract': 510_630}
]

M_rocket = 210292 # масса тела
g_kerb = 9.81 # ускорение свободного падения
R = 8.31 # универсальная газовая постоянная
M = 0.029 # молярная масса воздуха
p_0 = 101_325 # давление над уровнем моря в Па
start_temp = 300
rho_0 = 1.225  # плотность воздуха на уровне моря (кг/м³)
G = 6.67430e-11  # Гравитационная постоянная
M_kerbin = 5.2915793e22  # Масса Кербина в кг
R_kerbin = 600000  # Радиус Кербина в метрах
S = 13.56 # площадь поперечного сечения
C = 0.57 # коэффициент лобового спортивления
temperature = 300

def corn(altitude):
    if altitude < 80000:
        return 90 * (1 - altitude / 80000)
    return 0

def k(m, t):
    return m / t

def for_odeint(data, time, stage_ind):
    x, x_speed, y, y_speed = data
    global temperature
    cur_stage = stages[stage_ind]
    fuel_mass = cur_stage['fuel']
    F_traction = cur_stage['f_tract']
    burn_time = cur_stage['burn_time']
    k_i = k(fuel_mass, burn_time)
    if temperature > 30:
        temperature = start_temp - 6 * (y // 1000) # температура от высоты
    new_mass = M_rocket - k_i * time
    velocity = x_speed**2 + y_speed**2
    corner = corn(y)
    
    p = p_0 * np.exp((-g_kerb * y * M) / (R * temperature)) # давление от высоты
    RHO = (p * M) / (R * temperature) # плотность от высоты
    
    F_grav = (G * M_kerbin * new_mass) / ((R_kerbin + y)**2) # сила гравитации 
    F_resist = C * S * velocity * RHO / 2 # сила сопротивления
    
    x_acceleration = (F_traction - F_resist) * np.cos(np.radians(corner)) / new_mass # ускорение по оси Ох
    y_acceleration = ((F_traction - F_resist) * np.sin(np.radians(corner)) - F_grav) / new_mass # ускорение по оси Оу
    acceleration = np.sqrt(x_acceleration**2 + y_acceleration**2)
    
    dx = x_speed
    dx_speed = x_acceleration
    dy = y_speed
    dy_speed = y_acceleration
        
    return [dx, dx_speed, dy, dy_speed]

start_data = [0, 0, 0, 0]

# первая ступень
time_first_stage = np.linspace(0, stages[0]["burn_time"]) # Время работы первой ступени
result_first_stage = odeint(for_odeint, start_data, time_first_stage, args=(0,)) # Решение системы для первой ступени

# вторая ступень
M_rocket -= (stages[0]['only_stage'] + stages[0]['fuel']) # Масса ракеты после отсоединения первой ступени
time_second_stage = np.linspace(0, stages[1]["burn_time"], 100) # Время работы второй ступени
result_second_stage = odeint(for_odeint, result_first_stage[-1, :], time_second_stage, args=(1,)) # Решение системы для второй ступени

# Объединение результатов
time = np.concatenate([time_first_stage, time_first_stage[-1] + time_second_stage])
x = np.concatenate([result_first_stage[:, 0], result_second_stage[:, 0]])
x_speeds = np.concatenate([result_first_stage[:, 1], result_second_stage[:, 1]])
y = np.concatenate([result_first_stage[:, 2], result_second_stage[:, 2]])
y_speeds = np.concatenate([result_first_stage[:, 3], result_second_stage[:, 3]])
speeds = np.array([np.sqrt(x_speeds[i]**2 + y_speeds[i]**2) for i in range(len(x_speeds))])
math_accel = np.array([speeds[i] - speeds[i - 1] for i in range(1, len(speeds))])
math_accel_ox = np.array([x_speeds[i] - x_speeds[i - 1] for i in range(1, len(x_speeds))])
math_accel_oy = np.array([y_speeds[i] - y_speeds[i - 1] for i in range(1, len(y_speeds))])

# Считываем данные из КСП
with open("data_for_ksp.json", 'r', encoding='UTF-8') as file:
    data_ksp = json.load(file)
    
time_ksp = np.array(data_ksp[0]["pastime"])
altitude_ksp = np.array(data_ksp[0]["height"]) 
speed_ksp = np.array(data_ksp[0]["velocity"]) 
ox_speed_ksp = np.array(data_ksp[0]["ox_velocity"])
oy_speed_ksp = np.array([data_ksp[0]["oy_velocity"][x] for x in range(0, len(data_ksp[0]["oy_velocity"]), 13)]) 
acceleration_ksp = np.array([data_ksp[0]["acc"][x] for x in range(0, len(data_ksp[0]["acc"]), 13)])
ox_acceleration_ksp = np.array([data_ksp[0]["ox_ac"][x] for x in range(0, len(data_ksp[0]["ox_ac"]), 13)]) 
oy_acceleration_ksp = np.array([data_ksp[0]["oy_ac"][x] for x in range(0, len(data_ksp[0]["oy_ac"]), 13)]) 

#Строим графики
plt.figure(figsize=(15, 5))
# высота от времени
plt.subplot(1, 2, 1)
plt.plot(time, y , label='Высота мат модели', color='black')
plt.plot(time_ksp, altitude_ksp, label='Высота КСП', color='orange')

plt.title('Высота от времени')
plt.xlabel('Время, с')
plt.ylabel('Высота, м')
plt.legend()
plt.grid()
# скорость от времени 
plt.subplot(1, 2, 2)
plt.plot(time, speeds , label='Скорость мат модели', color='black')
plt.plot(time_ksp, speed_ksp, label='Скорость КСП', color='orange')
plt.title('Скорость от времени')
plt.xlabel('Время, с')
plt.ylabel('Скорость м/с')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(15, 5))
# горизонтальная скорость от времени
plt.subplot(1, 2, 1)
plt.plot(time, x_speeds , label='Горизонтальная скорость мат модели', color='black')
plt.plot(time_ksp, ox_speed_ksp, label='Горизонтальная скорость КСП', color='orange')
plt.title('Горизонтальная скорость от времени')
plt.xlabel('Время, с')
plt.ylabel('Скорость, м/с')
plt.legend()
plt.grid()
# вертикальная скорость от времени
plt.subplot(1, 2, 2)
plt.plot(time, y_speeds , label='Вертикальная скорость мат модели', color='black')
plt.plot(np.array([time_ksp[x] for x in range(0, len(time_ksp), 13)]), oy_speed_ksp, label='Вертикальная скорость КСП', color='orange')
plt.title('Вертикальная скорость от времени')
plt.xlabel('Время, с')
plt.ylabel('Скорость, м/с')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(15, 5))
# ускорение от времени
plt.subplot(1, 2, 1)
plt.plot(time[:len(time)-1], math_accel , label='Ускорение мат модели', color='black')
plt.plot(np.array([time_ksp[x] for x in range(0, len(time_ksp), 13)]), acceleration_ksp, label='Ускорение КСП', color='orange')
plt.title('Ускорение от времени')
plt.xlabel('Время, c')
plt.ylabel('Ускорение, м/с^2')
plt.legend()
plt.grid()

# ускорение по оси Ох от времени
plt.subplot(1, 2, 2)
plt.plot(time[:len(time)-1], math_accel_ox, label='Горизонтальное ускорение мат модели', color='black')
plt.plot(np.array([time_ksp[x] for x in range(0, len(time_ksp), 13)]), ox_acceleration_ksp, label='Горизонтальное ускорение КСП', color='orange')
plt.title('Ускорение по оси Ох от времени')
plt.xlabel('Время, c')
plt.ylabel('Ускорение, м/с^2')
plt.legend()
plt.grid()
plt.show()
# ускорение по оси Оу от времени
plt.plot(time[:len(time)-1], math_accel_oy, label='Вертикальное ускорение мат модели', color='black')
plt.plot(np.array([time_ksp[x] for x in range(0, len(time_ksp), 13)]), oy_acceleration_ksp, label='Вертикальное ускорение КСП', color='orange')
plt.title('Ускорение по оси Оy от времени')
plt.xlabel('Время, c')
plt.ylabel('Ускорение, м/с^2')
plt.legend()
plt.grid()
plt.show()
