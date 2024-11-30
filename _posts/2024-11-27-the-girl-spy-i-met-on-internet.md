---
layout: post
title:  "Repolishing My Machine Learning Skills: A Journey Back to Mastery"
date:   2024-11-27 09:00:00 +0000
categories: machine learning
slug: machine-learning
permalink: /machine-learning/:year/:month/:day/:slug/
---

## Haha Motive

Started another day just to polish my Machine Learning skills worked back three years ago , and i just went through the beginners course on machine learning from freecode camp.org ,  its said that Every  master was always a begginer ,haha for me my short memory makes it opposite and starting again from the begining 

#### Data Cleaning 

To make a data representable its necessary to make the data in its appropriate format to learn from the basics i have taken the data set [Dataset-link](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)  now its time to make things and data interesting so i just follow the instructions from freecodecamp.org and done the exercises in google colab notebook and uploaded the data set as a file to the the new project ,after that i openened the notebook and in a cell begin to write the magical codes 

some important libraries 
```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```

Now we have to define feature vectors(column headings) and give the data column names 


```python

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("sample_data/magic04.data", names=cols)
df.head()
#output looks like 
#	fLength	fWidth	fSize	fConc	fConc1	fAsym	fM3Long	fM3Trans	fAlpha	fDist	class
# 0	28.7967	16.0021	2.6449	0.3918	0.1982	27.7004	22.0110	-8.2027	    40.0920	81.8828	 g
# 1	28.7967	16.0021	2.6449	0.3918	0.1982	27.7004	22.0110	-8.2027	    40.0920	81.8828	 h 
```
After seeing this data we have an output label which have only two type g and h (gamma and hadron) this is obtained from analysing the feature vectors or input labels

For binary classification its necessary to represent output labels as 0 and 1 instead of g and h ,si to transform the output labels we have to write a piece of code in cell

```python
df["class"] = (df["class"] == "g").astype(int)
df.head()
#output 
#	fLength	fWidth	fSize	fConc	fConc1	fAsym	fM3Long	fM3Trans	fAlpha	fDist	class
# 0	28.7967	16.0021	2.6449	0.3918	0.1982	27.7004	22.0110	-8.2027	    40.0920	81.8828	 1
# 1	28.7967	16.0021	2.6449	0.3918	0.1982	27.7004	22.0110	-8.2027	    40.0920	81.8828	 0

```

For now at least we made our data some cleaner 

### What is Machine Learning : A summary 
 Lets see machine learning as a time machine making predictions based on some algorithms and previous data

### Types of ML
  *1.* **Supervised Learning**

  Used labeled inputs or feature vectors to train models and learn from the output labels . Always remember labeled input have labeled outputs

  *2.* **Unsupervised Learning**

  Unsupervised learning uses unlabeled data to identify patterns and structures. It often involves clustering, where data points with similar patterns are grouped together.example :
  a dataset of customer shopping behaviors at a grocery store:
**Age of the customer**,**Amount spent per visit**,**Frequency of visits**,**Preferred product categories(e.g., dairy, snacks, fruits)**.In this case, unsupervised learning (like clustering) could group customers into segments, such as:*Frequent spenders: Customers who visit often and spend a lot*.
,*Bargain hunters: Customers who visit rarely but purchase discounted products.*,
,*Category loyalists: Customers who focus on specific categories, like organic produce.* 

  
  *3.* **Reinforcement  Learning**: Based on rewards 



###  1. SUPERVISED LEARNING

As we discussed earlier we have a group of input labels passing therough models which will produce output labels known as Predictions 

I/P ->->-> [MODEL] ->->-> O/P

All inputs are feature vectors 

####  Types of feature Vectors 
1. **Qualitative** :
        These are categorical or descriptive features that represent qualities or categories, not measurable by numbers.
        
    Examples:  
     Gender: Male, Female, Other  
     Colors: Red, Green, Blue

	 Labels like “Yes” or “No”
    Qualitative features are often converted into numerical representations for machine learning models. This can be done through techniques like:

	• **One-Hot Encoding**: Creating binary columns for each category.  
       Example grouping countries [USA,INIDA,CANADA,FRANCE]  
       USA = 1000  
       INDIA = 0100  
       CANADA = 0010  
       FRANCE = 0001  
        
	• **Label Encoding**: Assigning numerical codes to categories (e.g., 0 for Male, 1 for Female).

2. **Quantitative** : These are numerical features that represent measurable quantities  like discrete or continous , temp,age,salary etc

####  Supervised Learning Tasks

1. **Classification**  
   In this the task is to predict output from a discrete class and this type of classification is again divided into Binary Claassification and multiclass classification, **Binary classification** which includes examples like **positive or negative results**,**images of cats and dogs**,**checking mail spam or not**,**Handwritten digit classification (0, 1, 2, …, 9)**

   ***multiclass classification** : **images of cat ,lizard and dog** ,**animal species**

2. **Regression**   : *To Predict Contiionous Value*  
   **price of btc prediction**,**temperature**,**price of house**

####  How models learn?

models learn from the data we gave , normally each **row** is a **different sample**   
**each column = differnt feature** without target label  feature vector all together is known as **feature matrix** and the target label is known as **target  output**  

Normally **feature matrix represented as X** and **Target output as Y**



####  Supervised Learning Datasets 

 supervised learning datasets datasets are typically classified into the following three categories to ensure the model is trained effectively and evaluated accurately:
 
 1. **Training Dataset**  
    The training dataset is used to train the model. It consists of input-output pairs (features and labels) that the model learns from.  

 2. **Vaidation Dataset**  
    Validation set used as a reality check during and after training to ensure model can handle unseen data

 3. **Testing Dataset**
    The testing dataset is used to evaluate the final model’s performance after training is complete. It measures how well the model generalizes to completely unseen data.

    ### Common Dataset Splits 
    **Training: ~70-80% of the data.**  
    **Validation: ~10-15% of the data.**  
    **Testing: ~10-15% of the data.**

####  What is loss in Machine Learning?
Loss is a metric that represents how well (or poorly) a machine learning model is performing during training. It quantifies the difference between the model’s predictions and the actual target values. A smaller loss indicates better model performance.

####  What is Loss function in machine learning ? 
A loss function (also called a cost function or objective function) is a mathematical function that measures how well the predictions made by a model align with the true values (i.e., the ground truth) in supervised learning. It quantifies the difference between the predicted output of the model and the actual output (real labels) and provides a signal to the learning algorithm on how to adjust the model’s parameters to improve its performance.  

**Types of Loss Function**

1. #### Mean Squared Error (MSE) - For Regression
For regression tasks, where the model predicts continuous values (e.g., predicting prices, temperature), a common loss function is the Mean Squared Error (MSE).  
     **loss=sum((y_real-y_predicted)^2)**

2. #### Mean Absolute  Error (MAE) - For Regression
Unlike MSE, MAE uses the absolute difference between the actual and predicted values, so it does not amplify large errors as much as MSE does. MAE is more robust to outliers than MSE.
     **loss=sum(|y_real-y_predicted|)**

3. #### Binary Cross-Entropy (Log Loss) - For Binary Classification
For binary classification (e.g., classifying whether an email is spam or not), the Binary Cross-Entropy loss function is commonly used. It calculates the error between the predicted probabilities and the actual binary labels (0 or 1).  
     **loss= -1/N * sum(y_real * log(y_predicted)) + (1-y_real) * log(1-y_predicted)**

# A  lot of blah blah now lets move to code

```python 
for label in cols[:-1]:
  plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
  plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()
  # just to plot the feature vectors using matplotlib 
  ```

  ``` python

  train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

  ```

1.	df.sample(frac=1):  
This shuffles the rows of the DataFrame df.  
The parameter frac=1 means all rows are included in the shuffle.

2. np.split():
This function splits an array (or DataFrame in this case) into multiple parts based on specified indices.

3.	[int(0.6*len(df)), int(0.8*len(df))]:
These are the indices at which the splits occur.  
int(0.6*len(df)): 60% of the dataset’s length, which determines the end of the training set.  
int(0.8*len(df)): 80% of the dataset’s length, which determines the end of the validation set.(from 60% to 80%).  
The remaining 20% forms the test set.

	•	Training Set: Used to train the model. The model learns patterns from this dataset.  
    •	Validation Set: Used to tune hyperparameters and evaluate the model during training without exposing it to the test data.  
    •	Test Set: Used for the final evaluation of the model after training is complete.

```python
def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))
  # means reshape into column vector
  # Reshaped to a column vector shape(rows and columns) [
  1
  0
  1
  #](3,1) 3 rows 1 col
  return data, X, y

```

#### Why Use StandardScaler?
It ensures all features are on the same scale (mean = 0, standard deviation = 1)preventing larger features from dominating smaller ones.It is essential for distance-based models and optimization algorithms.

#### What is RandomOverSampler?
The oversampling process involves creating synthetic examples of the minority class by randomly duplicating its samples. This results in a balanced dataset where both the majority and minority classes have a similar number of examples.  
    •	Input Dataset:  
    •	Majority Class: 900 samples (e.g., “No Fraud”)  
    •	Minority Class: 100 samples (e.g., “Fraud”)  
    •	After Applying RandomOverSampler:  
    •	Majority Class: 900 samples  
    •	Minority Class: 900 samples (by duplicating existing samples)

```python
train, X_train, y_train = scale_dataset(train, oversample=True)
# train = scaled data combining both x and y after oversampling
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)
```

### Models for Machine Learning
#### KNN Model (k neatest Neighbours)  
The K-Nearest Neighbors (KNN) algorithm is a simple yet powerful supervised learning method used for both classification and regression tasks. Its core idea is to make predictions based on the similarity of data points in the feature space.  

#### How KNN Works:
1.	Training Phase:
		KNN does not explicitly “train” a model. Instead, it stores the entire dataset in memory.This is called a lazy learning algorithm because it defers computation until prediction time.
2.	Prediction Phase:
	•	To classify or predict a new data point:
	1.	Calculate the distance (usually Euclidean) between the new point and all points in the training data.
	2.	Identify the K nearest neighbors (data points with the smallest distances).
3.	For classification:
	•	Assign the class that is most frequent among the K neighbors (majority voting).
4.	For regression:
	•	Calculate the average or weighted average of the target values of the K neighbors.

#### Key Parameters
1. **K (Number of Neighbors):**
	•	Controls how many neighbors are considered when making a prediction.  
    •	**Small  K** : Sensitive to noise and overfitting.  
    •	**Large  K** : Smoother decision boundary but may overlook fine details.
2.	**Distance Metric:**
	•	Common metrics include:  
    •	**Euclidean Distance**: Straight-line distance.  
	•	**Manhattan Distance:** Sum of absolute differences.  
    •	**Cosine Similarity**: Based on the angle between vectors.
3.	**Weighting of Neighbors:**
	•	All neighbors can be weighted equally, or closer neighbors can be given more influence.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knn_model = KNeighborsClassifier(n_neighbors=5)
#n_neighbors=5:
#Specifies that the model will consider the 5 nearest neighbors to classify each data point.
knn_model.fit(X_train, y_train)
#fit:This method trains the model using the training data (X_train) and their corresponding labels (y_train)
y_pred = knn_model.predict(X_test)
#y_predict:Predicts the labels for the test dataset (X_test) using the trained KNN model.

print(classification_report(y_test, y_pred))
#Compares the actual labels (y_test) with the predicted labels (y_pred) and calculates key metrics:
```
For each class (e.g., 0 and 1), you have three primary metrics:

**Precision:**Measures how many of the predicted positive labels were actually correct.  
**Recall (Sensitivity/True Positive Rate):** Measures how many of the actual positive labels the model correctly identified.  
**F1-Score:**The harmonic mean of precision and recall.It balances the trade-off between the two, especially useful when the dataset is imbalanced.