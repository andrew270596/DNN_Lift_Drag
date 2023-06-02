# Framework to create a deep neural network model which predicts lift and drag coefficents of 4-digits NACA airfoils 
# taking as input the NACA code and the angles of attack

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping

# Load the training data
from datasets_preparation import training_set

def main():
    inputs_lift = []
    inputs_drag = []
    outputs_lift = []
    outputs_drag = []
    # For each tuple in the training set:
    # * The digits of airfoil code are divided according to the airfoil characteristic they represent
    # * The data in the nested tuples are divided in 3 different lists, collecting respectively the AOAs,
    #   the lift coefficient and the drag coefficients
    for airfoil, data in training_set:
        max_camber = int(airfoil[4])
        max_camber_location = int(airfoil[5])
        max_thickness = int(airfoil[6:7])
        angles = [angle for angle, drag, lift in data]
        lifts = [lift for angle, drag, lift in data]
        drags = [drag for angle, drag, lift in data]

        # The variables/lists obtained from the training set are divided in input data or output control
        # data of the two multilayer percentrons constituting the deep neural network
        for i in range(len(angles)): 
            inputs_lift.append([max_camber, max_camber_location, max_thickness, angles[i]])
            inputs_drag.append([max_camber, max_camber_location, max_thickness, angles[i], lifts[i]])
            outputs_lift.append([lifts[i]]) 
            outputs_drag.append([drags[i]]) 


    # Architecture definition of the multilayer percetron predicting the lift coefficients:
    # * It has 6 layers: a input layer, 4 hidden layers and an output layer
    # * The nodes composing the MLP are arranged in configuration 4,32,16,16,16,1
    # * The artificial neurons have leaky ReLU as activation function, except on the output layer,
    #   where the identity function is adopted
    # * The first hidden layer has a L2 regularizer
    lift_model = tf.keras.Sequential()
    lift_model.add(tf.keras.layers.Dense(32, input_dim=4, activation=tf.keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer=regularizers.l2(l2=0.01)))
    lift_model.add(tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
    lift_model.add(tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
    lift_model.add(tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
    lift_model.add(tf.keras.layers.Dense(1))

    # The optimization during the training phase of the lift MLP is done using an ADAM optimizer 
    # on a mean squared error loss function
    opt = keras.optimizers.Adam(learning_rate=0.001)
    lift_model.compile(optimizer=opt, loss='mean_squared_error')

    # Declarration of the early stopping mechanism for the lift MLP. It stops the training phase of MLP in case the validation loss 
    # didn't reach a new minimum in the interval of 30 epochs
    trainingStopCallback_lift = EarlyStopping(monitor='val_loss', patience=30)

    # The training phase of the lift MLP is run: 
    # * In the first epoch the data are divided in training and validation sets in a ratio of 9:1
    # * The training occours for a maximum of 500 epochs
    lift_model.fit(np.array(inputs_lift), np.array(outputs_lift), validation_split=0.1, epochs=500, callbacks=[trainingStopCallback_lift])

    # Architecture definition of the multilayer percetron predicting the drag coefficients:
    # * It has 6 layers: a input layer, 4 hidden layers and an output layer
    # * The nodes composing the MLP are arranged in configuration 5,32,16,16,16,1
    # * The artificial neurons have leaky ReLU as activation function, except on the output layer,
    #   where the identity function is adopted
    # * The first hidden layer has a L2 regularizer
    drag_model = tf.keras.Sequential()
    drag_model.add(tf.keras.layers.Dense(64, input_dim=5, activation=tf.keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer=regularizers.l2(l2=0.01)))
    drag_model.add(tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
    drag_model.add(tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
    drag_model.add(tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
    drag_model.add(tf.keras.layers.Dense(1))

    # The optimization during the training phase of the drag MLP is done using an ADAM optimizer 
    # on a mean squared error loss function
    opt = keras.optimizers.Adam(learning_rate=0.001) 
    drag_model.compile(optimizer=opt, loss='mean_squared_error')

    # Declarration of the early stopping mechanism for the drag MLP. It stops the training phase of MLP in case the validation loss 
    # didn't reach a new minimum in the interval of 30 epochs
    trainingStopCallback_drag = EarlyStopping(monitor='val_loss', patience=30)

    # The training phase of the drag MLP is run: 
    # * In the first epoch the data are divided in training and validation sets in a ratio of 9:1
    # * The training occours for a maximum of 500 epochs
    drag_model.fit(np.array(inputs_drag), np.array(outputs_drag), validation_split=0.1, epochs=400, callbacks=[trainingStopCallback_drag])

    # The lift and drag MLPs generated are saved on hierarchical data format files
    lift_model.save("lift_model_new.h5")
    drag_model.save("drag_model_new.h5")

# Code that is run only if the file is run directly and not when it is imported
if __name__ == "__main__":
    main()