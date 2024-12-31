# Rock-or-Mine
## Description:
A machine learning classification model has been developed to predict whether an underwater object is a Rock or a Mine, enhancing object identification for submarines and underwater vehicles.

The process begins with sonar waves emitted underwater. These waves interact with submerged objects and reflect back, carrying information about the object's characteristics. The reflected signals are captured and processed to extract meaningful features.

Specialized laboratories transform the raw sonar data into a structured dataset used to train the model. This dataset includes acoustic features such as signal intensity, frequency, and waveform patterns, which help distinguish rocks from mines. By analyzing these features, the model learns to make accurate classifications.

The model prioritizes reliability, as accurate detection is critical in underwater operations. Misclassifications, such as identifying a mine as a rock, could lead to severe consequences. To ensure robustness, the model undergoes rigorous training and testing phases, with techniques applied to reduce overfitting and improve performance on unseen data.

This system automates the classification of sonar signals, reducing reliance on manual analysis and minimizing the risks of human error. By speeding up decision-making processes, it significantly enhances the safety and operational efficiency of submarines and other underwater vehicles.

Future improvements include refining the model with diverse datasets that account for real-world conditions, such as varying depths, salinity, and noise interference. Additionally, integrating the system with real-time onboard processing units could enable immediate object classification, further advancing underwater navigation and hazard detection.

This machine learning-based approach marks a significant step forward in underwater exploration and safety, contributing to the success and reliability of critical underwater missions.
## Code:
#### Importing Libraries
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
#### Data Collection and Processing
```
sonar_df=pd.read_csv('/Users/home/........../Sonar Data.csv',header=None)
```
```
sonar_df.head()
```
```
X=sonar_df.drop(columns=60)
Y=sonar_df[60]
print(X)
print(Y)
```
#### Training and Test Data Splitting
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
```
```
model=LogisticRegression()
model.fit(X_train,Y_train)
```
#### Model Evaluation [Training Data]
```
X_train_prediction=model.predict(X_train)
Training_Accuracy=accuracy_score(X_train_prediction,Y_train)
print(f"Accuracy on the Training Data is : {Training_Accuracy}")
print(f"Accuracy on the Training Data is : {round(Training_Accuracy*100,3)}%")
```
#### Model Evaluation [Test Data]
```
X_test_Prediction=model.predict(X_test)
Test_Accuracy=accuracy_score(X_test_Prediction,Y_test)
print(f"The Accuracy on Test Data is {Test_Accuracy}")
print(f"The Accuracy on Test Data is {round(Test_Accuracy*100,3)}")
```
#### Creating a Predictive System
```
# input data
input_data=(0.0323,0.0101,0.0298,0.0564,0.0760,0.0958,0.0990,0.1018,0.1030,0.2154,0.3085,0.3425,0.2990,0.1402,0.1235,0.1534,0.1901,0.2429,0.2120,0.2395,0.3272,0.5949,0.8302,0.9045,0.9888,0.9912,0.9448,1.0000,0.9092,0.7412,0.7691,0.7117,0.5304,0.2131,0.0928,0.1297,0.1159,0.1226,0.1768,0.0345,0.1562,0.0824,0.1149,0.1694,0.0954,0.0080,0.0790,0.1255,0.0647,0.0179,0.0051,0.0061,0.0093,0.0135,0.0063,0.0063,0.0034,0.0032,0.0062,0.0067)
input_as_npy_array=np.asanyarray(input_data)
input_reshaped=input_as_npy_array.reshape(1,-1)
prediction=model.predict(input_reshaped)
print(prediction)

if prediction[0]=='R':
    print("It's a Rock")
else:
    print("It's a Mine")
```
## Output:
<img width="1606" alt="Screenshot 2024-12-31 at 4 49 48 PM" src="https://github.com/user-attachments/assets/06f54015-1eef-42f3-992a-bfbee2f9e422" />

