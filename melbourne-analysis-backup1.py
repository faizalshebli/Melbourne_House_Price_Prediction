#import libraries
import pandas as pd

#read data source file
melbourne_data = pd.read_csv('/Users/faizalshebli/Documents/#FS2021/FS transformation/FS5T plan/FS learn ML via Kaggle/Data/melb_data.csv')

#data set exploratory
melbourne_data.head()

#remove missing value 'na'
melbourne_data = melbourne_data.dropna(axis=0)

#selecting the predeition target
y = melbourne_data.Price

#selecting the features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X=melbourne_data[melbourne_features]

#building my model

from sklearn.tree import DecisionTreeRegressor
#Define model. Specify a number for random_state to ensure same results each run

melbourne_model = DecisionTreeRegressor(random_state=1)

#Fit model
melbourne_model.fit(X, y)

DecisionTreeRegressor(random_state=1)

#printing the predictive model result

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
