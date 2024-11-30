# -*- coding: utf-8 -*-
from pyfirmata import Arduino, util
import numpy as np
import random
import time
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

board = Arduino('COM25')
it = util.Iterator(board)
it.start()

pin8 = board.get_pin('d:8:o')
pin9 = board.get_pin('d:9:o')
pin10 = board.get_pin('d:10:o')
pin11 = board.get_pin('d:11:o')
servo1 = board.get_pin('d:5:s')
servo2 = board.get_pin('d:6:s')
echo_pin = board.get_pin('d:7:o')

time.sleep(1)

def stop(t=0):
    pin8.write(0)
    pin9.write(0)
    pin10.write(0)
    pin11.write(0)
    if t: time.sleep(t)

def move_forward(t):
    pin10.write(0)
    pin11.write(1)
    pin8.write(0)
    pin9.write(1)
    time.sleep(t)
    stop()

def move_backward(t):
    pin10.write(1)
    pin11.write(0)
    pin8.write(1)
    pin9.write(0)
    time.sleep(t)
    stop()

def rotate_left(t):
    pin10.write(0)
    pin11.write(1)
    pin8.write(1)
    pin9.write(0)
    time.sleep(t)
    stop()

def rotate_right(t):
    pin10.write(1)
    pin11.write(0)
    pin8.write(0)
    pin9.write(1)
    time.sleep(t)
    stop()

def ping(n=3):
    "Повертає середнє значення відстані, виміряної ультразвуковим сенсором"
    return sum([util.ping_time_to_distance(echo_pin.ping()) for _ in range(n)]) / n

def scan_3d():
    angles = []
    distances = []
    for v in range(60, 131, 10):  
        servo2.write(v)
        time.sleep(0.05)
        for h in range(0, 181, 10): 
            servo1.write(h)
            time.sleep(0.1)
            dist = ping(5)
            angles.append((h, v))
            distances.append(dist)
    return angles, distances

def generate_training_data():
    X = []
    Y = []
    for _ in range(100):
        data = [random.random() for _ in range(5)] 
        X.append(data)
        Y.append(random.choice([0, 1]))  
    return np.array(X), np.array(Y)

def train_model(X, Y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  
    model = LogisticRegression()
    model.fit(X_scaled, Y)
    return model, scaler

def predict(model, scaler, data):
    data_scaled = scaler.transform([data])
    return model.predict(data_scaled)[0]


def navigate(model, scaler):
    while True:
        angles, distances = scan_3d()
        data = [int(d < 40) for d in distances]  
        p = predict(model, scaler, data) 
        if p == 1:
            print("Об'єкт виявлений. Поштовх!")
            move_forward(2)
        else:
            print("Об'єкта немає. Рух в випадковому напрямку.")
            move_random_direction()

def move_random_direction():
    direction = random.choice(['left', 'right', 'forward', 'backward'])
    if direction == 'left':
        rotate_left(1)
    elif direction == 'right':
        rotate_right(1)
    elif direction == 'forward':
        move_forward(1)
    else:
        move_backward(1)

X, Y = generate_training_data()
model, scaler = train_model(X, Y)

navigate(model, scaler)

board.exit()
