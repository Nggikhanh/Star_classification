import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from Star import Knearestneighbor



file_path = "D:/Uni works/Third year/Machine learning/Project/Final_processed4.csv"
stars_ohe = pd.read_csv(file_path)
scaler = StandardScaler()
stars_ohe_scaled = scaler.fit(stars_ohe.drop(['Star type','Index'], axis = 1))
stars_ohe_scaled = scaler.transform(stars_ohe.drop(['Star type','Index'], axis = 1))


cols = ['Temperature (K)',
        'Luminosity (L/Lo)',
        'Radius (R/Ro)',
        'Absolute magnitude (Mv)',
        'Color_Blue',
        'Color_Blue-White',
        'Color_Orange',
        'Color_Red',
        'Color_White',
        'Color_Yellow-White',
        'Class_A',
        'Class_B',
        'Class_F',
        'Class_G',
        'Class_K',
        'Class_M',
        'Class_O']


label_encoder = LabelEncoder()
X = pd.DataFrame(stars_ohe_scaled, columns = cols)
X = stars_ohe_scaled
Y = stars_ohe['Star type']

y = label_encoder.fit_transform(Y)
X = scaler.fit_transform(X)

# n_neighbor_values = range(1,21)
kf = KFold(n_splits=5, shuffle=True, random_state=45)
cross_val_scores = []
for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        knn = Knearestneighbor(X_train,Y_train,n_neighbors=5)


st.header("Star Type Classification:")
image = Image.open('D:/Uni works/Third year/Machine learning/Project/nasaimage.jpeg')
st.image(image, use_column_width=True)
st.write("Please insert values, to get Star type class prediction")


# Scaled sliders
Temperature_ip = st.slider('Temperature:', -0.89, 3.09)
L_ip = st.slider('Luminosity:', -0.59, 4.14, step=0.00001)
R_ip = st.slider('Relative Radius:', -0.45, 3.31)
A_M_ip = st.slider('Absolute Mass:', -1.55, 1.49)

# Convert scaled values back to the original ranges
temperature_ip_min, temperature_ip_max = 1939, 40000
temperature_ip_original = (Temperature_ip + 0.89) / (3.09 + 0.89) * (temperature_ip_max - temperature_ip_min) + temperature_ip_min

L_ip_min, L_ip_max = 0.00008, 849420.0
L_ip_original = (L_ip + 0.59) / (4.14 + 0.59) * (L_ip_max - L_ip_min) + L_ip_min

R_ip_min, R_ip_max = 0.0084, 1948.5
R_ip_original = (R_ip + 0.45) / (3.31 + 0.45) * (R_ip_max - R_ip_min) + R_ip_min

A_M_ip_min, A_M_ip_max = -11.92, 20.06
A_M_ip_original = (A_M_ip + 1.55) / (1.49 + 1.55) * (A_M_ip_max - A_M_ip_min) + A_M_ip_min

# Create a container to display the overlay content
with st.container():
    st.write("Real value    :")
    st.write(f"Temperature Value (Original Range): {temperature_ip_original:.2f} K")
    st.write(f"Luminosity Value (Original Range): {L_ip_original:.2f}")
    st.write(f"Relative Radius Value (Original Range): {R_ip_original:.2f}")
    st.write(f"Absolute Mass Value (Original Range): {A_M_ip_original:.2f} MV")

st.header("Color Selection")
Color_Blue_ip = st.radio("Is the color blue?", ["0", "1"])
Color_Blue_White_ip = st.radio("Is the color blue-white?", ["0", "1"])
Color_Orange_ip = st.radio("Is the color orange?", ["0", "1"])
Color_Red_ip = st.radio("Is the color red?", ["0", "1"])
Color_White_ip = st.radio("Is the color white?", ["0", "1"])
Color_White_Yellow_ip = st.radio("Is the color white-yellow?", ["0", "1"])
st.header("Spectral Class Selection")
Class_A_ip = st.radio("Is the spectral class A?", ["0", "1"])
Class_B_ip = st.radio("Is the spectral class B?", ["0", "1"])
Class_F_ip = st.radio("Is the spectral class F?", ["0", "1"])
Class_G_ip = st.radio("Is the spectral class G?", ["0", "1"])
Class_K_ip = st.radio("Is the spectral class K?", ["0", "1"])
Class_M_ip = st.radio("Is the spectral class M?", ["0", "1"])
Class_O_ip = st.radio("Is the spectral class O?", ["0", "1"])

data = {'Temperature': Temperature_ip,
        'Luminosity': L_ip,
        'Relative radius': R_ip,
        'Absolute magnitude': A_M_ip,
        'Color_Blue' : Color_Blue_ip,
        'Color_Blue_White' : Color_Blue_White_ip,
        'Color_Orange' : Color_Orange_ip,
        'Color_Red' : Color_Red_ip,
        'Color_White' : Color_White_ip,
        'Color_White_Yellow' : Color_White_Yellow_ip,
        'Class_A' : Class_A_ip,
        'Class_B' : Class_B_ip,
        'Class_F' : Class_F_ip,
        'Class_G' : Class_G_ip,
        'Class_K' : Class_K_ip,
        'Class_M' : Class_M_ip,
        'Class_O' : Class_O_ip
        }


# print("***"*30)
feature = pd.DataFrame(data, index = [0])
feature = feature.apply(pd.to_numeric, errors='coerce')
Input_data = feature.values.flatten()
Input_data = Input_data.reshape(1, -1)

kf = KFold(n_splits=5, shuffle=True, random_state=45)
cross_val_scores = []
for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        knn = Knearestneighbor(X_train,Y_train,n_neighbors=5)
        Predict2 = knn.predict_proba(Input_data)
        
mean_probabilities = np.mean(Predict2, axis=0)
mean_probabilities = mean_probabilities.reshape(1, -1)

Predict1 = knn.predict(Input_data)



st.subheader('Star type predict:')
st.write('**Star type**:',Predict1)

st.subheader('Prediction Percentages:')  
st.write('**Probablity of Star Type being 0 i.e. Brown Dwarf is ( in % )**:',mean_probabilities[0][0]*100)
st.write('**Probablity of Star Type being 1 i.e. Red Dwarf is ( in % )**:',mean_probabilities[0][1]*100)
st.write('**Probablity of Star Type being 2 i.e. White Dwarf is ( in % )**:',mean_probabilities[0][2]*100)
st.write('**Probablity of Star Type being 3 i.e. Main sequence is ( in % )**:',mean_probabilities[0][3]*100)
st.write('**Probablity of Star Type being 4 i.e. Super Giants is ( in % )**:',mean_probabilities[0][4]*100)
st.write('**Probablity of Star Type being 5 i.e. Hyper Giants is ( in % )**:',mean_probabilities[0][5]*100)

