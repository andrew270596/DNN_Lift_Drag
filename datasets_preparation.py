# The file reads the data contained in the airfoils_aerodynamic_coefficients.xlsx and arrange them in a ordered 
# list of tuples readable by the other file of the package

import openpyxl

def excel_to_training_set(filename):
  # Loading of the Excel workbook
  wb = openpyxl.load_workbook(filename) 
  # Acquisition the sheet with the airfoil data
  sheet = wb["Training Set"]  
  # Initialization of an empty training set
  training_set = []
  
  # For iteration through the rows in the sheet
  for row in sheet.rows:
    # Stops the reading of blank rows
    if row[0].value is None: break
    # Acquisition of the NACA airfoil code and angle of attack from the first two cells
    airfoil = row[0].value
    angle = row[1].value
    # Acquisition of the lift and drag coefficients from the remaining cells and round them up to 5 decimals
    lift = round(row[2].value,5)
    drag = round(row[3].value,5)   
    found = False
    for a in training_set:
      # Check if the current airfoil is already in the training set
      if a[0] == airfoil:
        # In case the current airfoil is found, the program adds a nested tuple with the angle and aerodynamic coefficients
        # to the tuple containing the current airfoil
        a[1].append((angle, lift, drag))
        found = True
        break  
    # In case the current airfoil was not found among the airfoils already in the training set, a new tuple is added to 
    # the training_set list. This will contain the airfoil code and the nested tuple with the angle and the aerodynamic coefficients  
    if not found:
      training_set.append((airfoil, [(angle, lift, drag)]))

  # The nested tuples in each tuple of the training_set list are ordered by increasing angle of attack
  for i, (airfoil, data) in enumerate(training_set):
    sorted_data = sorted(data, key=lambda x: x[0])
    training_set[i] = (airfoil, sorted_data)

  return training_set


def excel_to_testing_set(filename):
  # Loading of the Excel workbook
  wb = openpyxl.load_workbook(filename) 
  # Acquisition the sheet with the airfoil data
  sheet_testing = wb["Testing Set"]  
  # Initialization of an empty testing set
  testing_set = []

  # For iteration through the rows in the sheet
  for row in sheet_testing.rows:
    # Stops the reading of blank rows
    if row[0].value is None: break
    # Acquisition of the NACA airfoil code and angle of attack from the first two cells
    airfoil = row[0].value
    angle = row[1].value
    # Acquisition of the lift and drag coefficients from the remaining cells and round them up to 5 decimals
    lift = round(row[2].value,5)
    drag = round(row[3].value,5)  
    found = False
    for a in testing_set:
      # Check if the current airfoil is already in the testing set
      if a[0] == airfoil:
        # In case the current airfoil is found, the program adds a nested tuple with the angle and aerodynamic coefficients
        # to the tuple containing the current airfoil
        a[1].append((angle, lift, drag))
        found = True
        break
    # In case the current airfoil was not found among the airfoils already in the testing set, a new tuple is added to 
    # the testing_set list. This will contain the airfoil code and the nested tuple with the angle and the aerodynamic coefficients
    if not found:
      testing_set.append((airfoil, [(angle, lift, drag)]))

  # The nested tuples in each tuple of the testing_set list are ordered by increasing angle of attack
  for i, (airfoil, data) in enumerate(testing_set):
    sorted_data = sorted(data, key=lambda x: x[0])
    testing_set[i] = (airfoil, sorted_data)

  return testing_set


# Test the functions
training_set = excel_to_training_set("airfoils_aerodynamic_coefficients.xlsx")
testing_set = excel_to_testing_set("airfoils_aerodynamic_coefficients.xlsx")


