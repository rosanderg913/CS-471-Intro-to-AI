import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


training_data = pd.read_csv("./training.csv", header=None, usecols=[19,23], names=['Time','Current'])
test_data = pd.read_csv("./test.csv", header=None, usecols=[0, 4], names=['Time','Current'])

training_data = training_data[training_data['Time'] <= 5.4]
test_data = test_data[test_data['Time'] <= 2.4]

import matplotlib.pyplot as plt

df = training_data
fault_start = 5.1
fault_end = 5.4
# Separate the data points
fault_data = df[(df['Time'] >= fault_start) & (df['Time'] <= fault_end)]
normal_data = df[(df['Time'] < fault_start) | (df['Time'] > fault_end)]

plt.figure(figsize=(40, 6))

# Plotting for column D
plt.scatter(normal_data['Time'], normal_data['Current'], c='blue', label='Normal Operation (D)', alpha=0.5)
plt.scatter(fault_data['Time'], fault_data['Current'], c='red', label='Fault (D)', alpha=0.5)


plt.title('Scatter Plot of Current Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

df = test_data
fault_start = 2.1
fault_end = 2.4
# Separate the data points
fault_data = df[(df['Time'] >= fault_start) & (df['Time'] <= fault_end)]
normal_data = df[(df['Time'] < fault_start) | (df['Time'] > fault_end)]

plt.figure(figsize=(40, 6))

# Plotting for column D
plt.scatter(normal_data['Time'], normal_data['Current'], c='blue', label='Normal Operation (D)', alpha=0.5)
plt.scatter(fault_data['Time'], fault_data['Current'], c='red', label='Fault (D)', alpha=0.5)


plt.title('Scatter Plot of Current Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

# Define segmenting and labeling function

def segment_labeling(data, window, overlap, time1, time2):

  # Define the number of data points per segment = window size

  #index determines the start of a window
  #in each step of segmenting loop
  index = 0

  #windolap incorporates overlaping percentage
  windolap = math.floor (window * overlap)

  # Create an empty DataFrame for storing the labels
  labels_df = pd.DataFrame(columns=['label'])

  time_series = []

  while (index + window) < len(data):
      # Extract a segment of data
      segment = data.iloc[index : (index+window)]

      # Labeling based on a given time (the oscillation time is given)
      if any((time1 <= t <= time2) for t in segment['Time']):
        label = 'oscillation'
      else:
        label = 'normal'

      time_series.append(segment['Current'])

      # Append the label to the labels DataFrame
      labels_df = pd.concat([labels_df, pd.DataFrame({'label': [label]})], ignore_index=True)

      #Shifting the index forward by stride = window - windolap
      index += window - windolap

  # return lables_df as a DataFrame
  return time_series, labels_df

window = 200
overlap = 0.75

train_X, train_y = segment_labeling(training_data, window, overlap, 5.1, 5.4)
test_X, test_y = segment_labeling(test_data, window, overlap, 2.1, 2.4)

train_y.value_counts()

test_y.value_counts()

X_train = np.array(train_X)
X_test = np.array(test_X)

print(X_train.shape)
print(X_test.shape)

# Step 1: Fit a machine learning algorithm of your choice to the training data
# I will be using a support vector machine (SVM) for this example
from sklearn.svm import SVC

# Create an instance of the SVM model
svm = SVC(random_state=42)

# Ensure y is a 1D array
train_y = train_y.values.ravel()
test_y = test_y.values.ravel()

# Fit the model to the training data
svm.fit(X_train, train_y)

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}

# Create an instance of the GridSearchCV class
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')

# Fit the model to the training data
grid.fit(X_train, train_y)

# Print the best hyperparameters
print(grid.best_params_)

# Plot a chart of the hyperparameter tuning process


# Step 3: Make predictions on the test data
final_model = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], random_state=42, class_weight='balanced')
final_model.fit(X_train, train_y)

# Make predictions on the test data
predictions = final_model.predict(X_test)

# Step 4: Evaluate the model
from sklearn.metrics import classification_report

# Print the classification report
print(classification_report(test_y, predictions))





