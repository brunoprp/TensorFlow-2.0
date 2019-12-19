#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:25:23 2019

@author: bruno
"""
#%%
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
#%% Load data base
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
plt.imshow(x_train[5])
#%%
# Images normalizer
x_train = x_train / 255.0
x_test = x_test / 255.0 

# Reshape vector images for MLP One dimensional
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
#%% Criate model arquiteture
class Network():
    def __init__(self, input_shape, number_class ):
        self.model = tf.keras.models.Sequential()
        self.input_shape = input_shape
        self.number_class = number_class
    
    def net(self):
        self.model.add(tf.keras.layers.Dense(units = 128, activation = 'relu', input_shape=(self.input_shape)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units = self.number_class, activation = 'softmax'))
        self.model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
        return self.model
        
#%% Instantiating the Neural Network
input_shape = (784, )
number_class = 10
obj_net = Network(input_shape, number_class)
network = obj_net.net()
network.summary()   
#%% Training the  neural Network
network.fit(x_train, y_train, epochs =30)

#%%  Test model data base test
test_loss, test_accuracy = network.evaluate(x_test, y_test)

#%% Save model
model_json = network.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)

#%% Save weights
network.save_weights("fashion_model.h5")