import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle

# Loads the dataset
input_file = "iris.csv"

# Loads the data array and the category array
train_data = np.loadtxt(input_file, delimiter=',', usecols={0,1,2,3}, skiprows=1)
train_result = np.loadtxt(input_file, dtype='str', delimiter=',', usecols={4}, skiprows=1)

# Split train and test 
X_train, X_test, y_train, y_test = train_test_split(train_data, train_result, test_size=0.25, random_state=10)

# Create a KNN classifier
classifier = KNeighborsClassifier(n_neighbors=2)

# Train
classifier = classifier.fit(X_train, y_train)

# Predict 
result = classifier.predict(X_test)

# Compute accuracy
accuracy = metrics.accuracy_score(y_test, result)
print(f"Accuracy {accuracy}")

# Show the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, result)
print(conf_matrix)

# Saves the classifier to disk
output_file = "classifier.sav"
pickle.dump(classifier, open(output_file, 'wb'))