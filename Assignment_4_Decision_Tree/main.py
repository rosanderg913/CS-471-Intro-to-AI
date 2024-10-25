from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

iris = load_iris()

# Split the data into features and labels
X = iris.data
y = iris.target

# Split the data into training and test sets using 80/20 split
training_data, test_data, training_labels, test_labels = train_test_split(X, y, test_size=0.2)

# Split the data into training and validation sets using 80/20 split
# The 2 hyperparameters I will be tuning are max_depth and min_samples_split
validation_training_data, validation_testing_data, validation_training_labels, validation_testing_labels = train_test_split(training_data, training_labels, test_size=0.2)

# Create a decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier
clf = clf.fit(training_data, training_labels)

# Create sets of hyperparameters to test
max_depth_list = [1, 2, 3, 4, 5]
min_samples_split_list = [2, 3, 4, 5, 6]

results = []

# Create a dictionary to store the scores
scores = {}
best_accuracy = 0

# Loop through each possible hyperparameter, checking the accuracy and updating our dictionary of 
#   optimal parameters if a new max in accuracy is found
for depth in max_depth_list:
    for sample in min_samples_split_list:
        new_clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=sample)
        new_clf = new_clf.fit(validation_training_data, validation_training_labels)
        validation_label_prediction = new_clf.predict(validation_testing_data)
        new_accuracy = accuracy_score(validation_testing_labels, validation_label_prediction)
        print('min_sample: ', sample, 'max_depth: ', depth, 'accuracy: ',  new_accuracy)
        results.append((depth, sample, new_accuracy))

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
            scores = {'Optimal Max Depth': depth, 'Optimal Min Samples Split': sample}

print('Best hyperparameters found: ', scores)

# Plot the results where x is max_depth and y is accuracy. have 'min_sample' number of lines of differing color
for sample in min_samples_split_list:
    x = [result[0] for result in results if result[1] == sample]
    y = [result[2] for result in results if result[1] == sample]
    plt.plot(x, y, label='min_samples_split = ' + str(sample))


# Print the classification report for the Decision Tree using the optimal hyperparameters
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy for Different Hyperparameters')
plt.legend()
plt.grid(True)
plt.show()

# Fit a tuned Decision Tree using optimal hyperparameters with the entire training data, and test it on the unseen testing data
clf = tree.DecisionTreeClassifier(max_depth=scores['Optimal Max Depth'], min_samples_split=scores['Optimal Min Samples Split'])
clf = clf.fit(training_data, training_labels)
# Predict the testing data using the tuned Decision Tree and print the classification report
print('Iris classification using the tuned sklearn Decision Tree:\n')
final_prediction = clf.predict(test_data)
print(classification_report(test_labels, final_prediction, target_names=iris.target_names))

# Plot the final Decision Tree
plt.figure(figsize=(10, 6))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, fontsize=8)
plt.title('Decision Tree Trained on Iris Dataset')
plt.show()
