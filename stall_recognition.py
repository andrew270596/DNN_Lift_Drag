# Mechanism for post-stall point predictions avoidance

import numpy as np
import tensorflow as tf

# Calculates the percentage error of value1 on value2
def calculate_percentage_error(value1, value2):
    percentage_error = 100 * abs(abs(value1) - abs(value2)) / abs(value2)
    return percentage_error

# Empirical method that defines whether the input predicitions were made in the airfoil's post-stall condition or not
def stall_recognition(airfoil, angle, predicted_drag, predicted_lift, drag_model, lift_model):
    # Lift and drag coefficent predictions are made at 2° of inclination lower of the input predictions
    predicted_lift_prev_ang = lift_model.predict(np.array([[int(airfoil[4]), int(airfoil[5]), int(airfoil[6:7]), (angle-2)]]))
    predicted_drag_prev_ang = drag_model.predict(np.array([[int(airfoil[4]), int(airfoil[5]), int(airfoil[6:7]), (angle-2), float(predicted_lift_prev_ang)]]))   
    # The program returns a suspected post-stall situation message whenever the predicted 
    # drag and lift coefficients at "angle" and "angle-2" differ for more than 15% and less than 8% respectively
    if calculate_percentage_error(predicted_drag_prev_ang, predicted_drag) > 15 and calculate_percentage_error(predicted_lift_prev_ang, predicted_lift) < 8:
        return True
    else:
        return False


def main():
    print(__file__, " run as main\n")


# Code that is run only if the file is run directly and not when it is imported
if __name__ == "__main__":
    main()