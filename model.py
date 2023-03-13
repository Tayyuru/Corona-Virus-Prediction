import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Loading dataset")
corona_data = pd.read_csv("data.csv")

print("Loaded Dataset")
print(corona_data.head())

#Loading the data into numpy arrays
X = corona_data.iloc[:,1:8].values
Y = corona_data.iloc[:, 8].values

#Splitting the test and train data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#Scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the model
log_reg = LogisticRegression(penalty = 'l2')

print("Started training")
log_reg.fit(X_train,Y_train)

print("Model training completed successfully")

#Model evaluation
print("******* Evaluating Model *********")

Y_pred = log_reg.predict(X_test)

#printing confusion matrix
confusion_matrix = confusion_matrix(Y_test,Y_pred)
print("Confusion matrix: ")
print(confusion_matrix)

accuracy_test = accuracy_score(Y_test,Y_pred)
print("Accuracy obtained: ", accuracy_test)

f1_score_test = f1_score(Y_test,Y_pred)
print("F1 score obtained: ", f1_score_test)

#Saving the model into pickle file
pickle.dump(log_reg,open("model.pkl","wb"))