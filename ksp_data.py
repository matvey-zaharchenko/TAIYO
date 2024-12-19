import krpc 
import time
import json

conn = krpc.connect(name='Luna-17')
vessel = conn.space_center.active_vessel

st_time = conn.space_center.ut

data = [{
    "pastime": [],
    "height": [],
    "velocity": [],
    "ox_velocity": [],
    "oy_velocity": [],
    "acc": [],
    "ox_ac": [],
    "oy_ac": []
}
]

prev_speed = 0
prev_Oxs = 0
prev_Oys = 0
while True:
    cur_time = conn.space_center.ut # текущее время
    
    past_time = cur_time - st_time # прошедшее время с начала полета
    
    altitude = vessel.flight().mean_altitude # берем высоту
    
    if altitude >= 80000: # проверка выхода на орбиту
        break
    
    speed = vessel.flight(vessel.orbit.body.reference_frame).speed # скорость КА
    acceleration = (speed - prev_speed) / 0.1 # ускорение КА
    prev_speed = speed
    
    Ox_speed = vessel.flight(vessel.orbit.body.reference_frame).horizontal_speed # скорость относительно оси Ох
    ox_acc = (Ox_speed - prev_Oxs) / 0.1 # ускорение относительно оси Ох
    prev_Oxs = Ox_speed
    
    Oy_speed = vessel.flight(vessel.orbit.body.reference_frame).vertical_speed # скорость относительно оси Оу
    oy_acc = (Oy_speed - prev_Oys) / 0.1 # ускорение относительно оси Оу
    prev_Oys = Oy_speed
    
    data[0]["pastime"] += [past_time]
    data[0]["height"] += [altitude]
    data[0]["velocity"] += [speed]
    data[0]["ox_velocity"] += [Ox_speed]
    data[0]["oy_velocity"] += [Oy_speed]
    data[0]["acc"] += [acceleration]
    data[0]["ox_ac"] += [ox_acc]
    data[0]["oy_ac"] += [oy_acc]
    
    time.sleep(0.1)
    
with open("data_for_ksp.json", 'w', encoding="UTF-8") as file: # запись данных в файл
    json.dump(data, file, ensure_ascii=False, indent=2)
    