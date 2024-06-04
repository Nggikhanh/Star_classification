import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Star import Knearestneighbor
from sklearn.preprocessing import LabelEncoder

file_path = "D:/Uni works/Third year/Machine learning/Project/Classification_Star_DB/DB/Stars.csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.dtypes)
print("Missing values: ")
print(df.isnull().sum()) #No missing values
print("Dulicating values: ")
print(df.duplicated().sum()) #No duplicated values
print("----"*30)

df = df.drop('Star category', axis=1)
# Find the correlation between the data
no_object = df.drop(['Spectral Class', 'Star color'], axis=1)
corr=no_object.corr()
sns.heatmap(corr,annot=True)
plt.show()
sns.countplot(x='Star type', data = df, palette='viridis')
plt.show()



color_mappings = {
    'Blue White': 'Blue-White', 'Blue white': 'Blue-White', 'Blue-White': 'Blue-White', 'Blue white ': 'Blue-White',
    'white': 'White', 'Whitish': 'White', 'Blue-white':'Blue-White','yellow-white':'Yellow-White',
    'Yellowish White': 'Yellow-White', 'yellowish': 'Yellow-White', 'Yellowish': 'Yellow-White', 'White-Yellow': 'Yellow-White',
    'Pale yellow orange': 'Orange', 'Orange-Red': 'Orange', 'Blue ': 'Blue' }


df['Star color'] = df['Star color'].map(color_mappings).fillna(df['Star color'])

for i in ['Star color', 'Spectral Class', 'Star type']:
  print("\n",i, " --->\n", df[i].unique())
  print("\n",i, " --->\n", df[i].value_counts())
  
label_encoder = LabelEncoder()
scaler = StandardScaler()


df1 = pd.concat([df, pd.get_dummies(df['Star color'], prefix='Color')], axis=1)
df1 = pd.concat([df1, pd.get_dummies(df['Spectral Class'], prefix='Class')], axis=1)
df1 = df1.drop(['Star color', 'Spectral Class'], axis=1)

# print(df1)

#Store new columns to the old data
df1 = df1[['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Absolute magnitude (Mv)','Color_Blue', 'Color_Blue-White',
       'Color_Orange', 'Color_Red', 'Color_White', 'Color_Yellow-White','Class_A',
       'Class_B', 'Class_F', 'Class_G','Class_K', 'Class_M', 'Class_O', 'Star type']]

scaled_features = scaler.fit_transform(df1.drop('Star type', axis=1)) #Standardizing data
df_feat = pd.DataFrame(scaled_features, columns=df1.columns[:-1]) 
# print(df_feat)
Y = df1['Star type']
X = df_feat

y = label_encoder.fit_transform(Y)
X = scaler.fit_transform(X)


n_neighbor_values = range(1,40,2)
kf = KFold(n_splits=5, shuffle=True, random_state=45)
error_rate = []
mean_accuracy = []
mean_accuracy_all = []
for n in n_neighbor_values:
    cross_val_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        knn = Knearestneighbor(X_train,Y_train,n_neighbors=n)
        predict = knn.predict(X_test)
        accuracy = accuracy_score(Y_test, predict)
        cross_val_scores.append(accuracy)
        probabiltity = knn.predict_proba(X_test)
        
        
        
    mean_accuracy = np.mean(cross_val_scores)    
    mean_accuracy_all.append(np.mean(cross_val_scores))
    mean_probabilities = np.mean(probabiltity, axis=0)
    
    


print(mean_probabilities)
print(mean_accuracy_all)
my_integer = 1
Error = [my_integer - x for x in mean_accuracy_all]
print(Error)
plt.figure(figsize = (10,6))
plt.plot(range(1,40,2),mean_accuracy_all,color = 'blue',linestyle = '--',marker = 'o',markerfacecolor='red',markersize = 10)
plt.title('Mean accuracy vs K value')
plt.ylabel('Mean accuracy')
plt.xlabel('K')
plt.show()

k_range = range(1,40,2)
plt.figure(figsize = (10,6))
plt.plot(k_range, Error, color = 'blue',linestyle = '--',marker = 'o',markerfacecolor='red',markersize = 10)
plt.title('Error1 Rate vs K')
plt.xlabel('Value of K for KNN')
plt.ylabel('Error rate')
plt.show()

