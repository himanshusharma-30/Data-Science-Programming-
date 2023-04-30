import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

#import os
#exit(os.getcwd()) 

svm_model = pickle.load(open('C:/Users/himan/Downloads/svm_model.pkl', "rb"))

print("\n*****************************************************")
print("* The USF Lawn Mover Owner Prediction Model *")
print("*****************************************************\n")
income = float(input("Enter your Income: "))
lotsize = float(input("Enter your Lotsize: "))

df = pd.DataFrame({'Income': [income],'Lot_Size': [lotsize] })
result = svm_model.predict(df)
probability = svm_model.predict_proba(df)

print(f"\n The USF Lawn Mover Owner Prediction Model for being lawnmover owner is {result[0]} with a probality : {probability[0][1]:.4f}.\n")
