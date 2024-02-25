# classification decision trees 
# data processing
# importing the libraries  
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
import sklearn as sk
# importing the data set 
dataset = pd.read_csv('Wholesale customers data.csv')

dataset.shape

dataset.head()

dataset.info()

dataset.describe(include='all')

sns.histplot(dataset.Fresh)
plt.title('annual spending')
plt.show()

plt.figure(figsize=(10,5))
plt.title("annual spending on fresh produce")
sns.histplot(x="Fresh", hue="Channel", data=dataset)
plt.show()

# decision  tree classification
# Determining the class feature and input features
X = dataset.iloc[:, [2, 3 , 4, 5, 6, 7]].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# scaling the training data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_s=s=sc.fit_transform(X_train)
X_test_s=sc.transform(X_test)

# fitting decision  tree classification to the training set 
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion= 'entropy', random_state = 0)
classifier.fit(X_train_s, y_train)

# evaluating the model
# predicting the test set results 
y_pred=classifier.predict(X_test_s)
print(y_pred)

print(y_test)

from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print('Confusing Matrix:')
print(cm,'\n\n')
print('____________________________________________________')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)


ax = sns.heatmap(cm, cmap='flare',annot=True, fmt='d')
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.title("Confusion Matrix", fontsize=12)
plt.show()

 
