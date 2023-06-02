# Program that, reading through all the airfoils in the training and testing sets + some user added 
# geometries, predicts the airfoil and the configuration that returns the maximum lift-to-drag ratio

import numpy as np
import tensorflow as tf

from datasets_preparation import training_set, testing_set
import stall_recognition as SR

# Import of the multilayer perceptrons subunits constituting the deep neural network
lift_model = tf.keras.models.load_model("lift_model.h5")
drag_model = tf.keras.models.load_model("drag_model.h5")


def main():
    naca_codes = []
    # List of airfoils added by the user
    naca_codes = ["NACA2406", "NACA3412", "NACA4312", "NACA4420", "NACA6406", "NACA7409"]
    combined_sets = training_set + testing_set
    for airfoil, data in combined_sets:
        # Addition of all the airfoils in the training and testing sets to the user added ones, avoiding repetitions
        if airfoil not in naca_codes:  
            naca_codes.append(airfoil)
    max_lift_to_drag_ratio = 0
    max_naca_code = None
    max_angle = None
    # Creation of a list of 30 equally distanced float values between -4 and 16
    angles = np.linspace(-4, 16, num=30)
    
    for airfoil in naca_codes:
        lift_coefficients = []
        drag_coefficients = []
        for alpha in angles:
            # For each airfoil in the naca_codes list, at each of the inclinations present in the angles list, the 
            # neural network is run to predict the lift and drag coefficents
            lift = lift_model.predict(np.array([[int(airfoil[4]), int(airfoil[5]), int(airfoil[6:7]), alpha]]))
            drag = drag_model.predict(np.array([[int(airfoil[4]), int(airfoil[5]), int(airfoil[6:7]), alpha, float(lift)]]))
            # The stall recognition mechanism is summoned in order to check whether the airfoil is in sub-stall
            # conditions at the given angle of attack
            suspected_stall = SR.stall_recognition(airfoil, alpha, drag, lift, drag_model, lift_model)
            # In case a post-stall condition is not detected, the predicted lift and drag values are aggregated to the
            # lift_coefficents and the drag_coefficents lists
            if not suspected_stall:
                lift_coefficients.append(lift)
                drag_coefficients.append(drag)
            else:
                break
        # The lift-to-drag ratio is calculated for each of the sub-stall angles of attack of the current airfoil,
        # the highest of these values is then assigned to the variable airfoil_max_lift_to_drag_ratio
        lift_to_drag_ratios = [lift/drag for lift, drag in zip(lift_coefficients, drag_coefficients)]
        airfoil_max_lift_to_drag_ratio = max(lift_to_drag_ratios)
        # In case the maximum L/D ratio of the current airfoil is greater than the values of all the airfoils analyzed previously
        # such value is recorded as the new global maximum. The code of the current airfoil and the angle at which the max L/D value 
        # was found are also saved in different variables
        if airfoil_max_lift_to_drag_ratio > max_lift_to_drag_ratio:
            max_lift_to_drag_ratio = airfoil_max_lift_to_drag_ratio
            max_naca_code = airfoil
            # The function inside the squared brachets finds the location of the maximum L/D on the list with all the L/D ratios of 
            # the airfoil. Such index corresponds, however, also to the location of the AOA at which that value was achieved, 
            # thus the function returns the "max angle"
            max_angle = angles[lift_to_drag_ratios.index(airfoil_max_lift_to_drag_ratio)]
    
    print("\nNACA code with maximum lift-to-drag ratio:", max_naca_code)
    print("Angle of attack at which the ratio peaks:", max_angle, "\nwith a value of:", max_lift_to_drag_ratio, "\n")


# Code that is run only if the file is run directly and not when it is imported
if __name__ == "__main__":
    main()
