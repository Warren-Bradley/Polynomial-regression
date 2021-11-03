import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import random


"""
================================
FUNCTIONS:
================================
"""

'''
call_data method

This method reads the csv file containing the data to be clustered and returns it as a list

param file_name String containing the name of the csv file to be read

return data_list List of the relevant information on inflation vs time
'''

def call_data(file_name):

	#data store as a tuple (year, inflation of that year) added into a list
	with open(f'{file_name}.csv','r') as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    line_count = 0
	    data_list = []

	    #for each row in the file, add the country and data as tuple to the list
	    for row in csv_reader:	    	
	    	       
	    	if line_count == 0:            
	        	line_count += 1
	    	else:
	    		
	    		if len(row)>0:

	    			data_list.append((int(row[0]),float(row[1])))
	        		
	    		
	    		line_count += 1	    		

	return data_list


"""
================================
PROGRAM IMPLEMENTATION:
================================
"""

#Calling reading and saving the relevant data file
#retrieving the information from the entry boxes	
data = call_data('South Africa - Inflation over time')

#Randomising the data set so that the training and test sets are mixed
random.shuffle(data)

#Creating the x and y sets from the dataset
data_X = []
data_Y =[]
for row in data:
	data_X.append(row[0])
	data_Y.append(row[1])

#creating the training data
x_train = np.array(data_X[:-10]).reshape(-1, 1)
y_train = np.array(data_Y[:-10]).reshape(-1, 1)

#creating the testing data
x_test = np.array(data_X[-10:]).reshape(-1, 1)
y_test = np.array(data_Y[-10:]).reshape(-1, 1)

#setting the degree of polynomial to model with
poly = PolynomialFeatures(degree = 3)

#Creating the polynomial x test and training sets
X_train_poly , X_test_poly = poly.fit_transform(x_train), poly.fit_transform(x_test)

#Creating the linear regression model using the poly x sets
model = LinearRegression()
model = model.fit(X_train_poly, y_train)

#finding the coeficient and intercetion values
coeff = model.coef_[0]
inter = model.intercept_

#Creating the regression equation
x_axis = np.arange(1960,2020,0.1)
regr = inter + coeff[1]*x_axis+coeff[2]*x_axis**2+coeff[3]*x_axis**3

#checking the accuracy of the prediction model
prediction = model.predict(X_test_poly)
r2 = r2_score(prediction, y_test)

#Displaying the R^2 fit value
print(r2)

#Plotting the data
plt.scatter(x_train,y_train, color = 'b', label = 'Training data')
plt.scatter(x_test,y_test, color = 'g', label = 'Testing data')
plt.plot(x_axis, regr, color = 'r', label = 'Regression')

#Creating the graph titles and grid
plt.title('SA inflation rate over time')
plt.xlabel('Year')
plt.ylabel('Inflation')
plt.legend()
plt.grid(True)

plt.show()