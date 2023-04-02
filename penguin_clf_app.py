## Penguin Classifiction application by github.com/tushar2704

# Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


#  Streamlit app body

st.write("""# 🐧Penguin Classifier🐧""") 
st.write("""#### by Tushar Aggarwal""")
st.write(" This app predicts the Palmer Penguin species")


st.sidebar.header("Please select features")

# Collects user inputs features & converts to DF
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type =["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island =st.sidebar.selectbox("Island",('Biscoe', 'Dream', 'Torgersen'))
        sex=st.sidebar.selectbox("Sex", ('male', 'female'))
        bill_length_mm =st.sidebar.slider("Bill length (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm=st.sidebar.slider("Bill depth (mm)", 13.1, 21.5,17.2)
        flipper_length_mm=st.sidebar._slider("Flipper length (mm)", 172.0,231.0, 201.0)
        body_mass_g = st.sidebar.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
        data={'island':island,
              'bill_length_mm':bill_length_mm,
              'bill_depth_mm':bill_depth_mm,
              'flipper_length_mm':flipper_length_mm,
              'body_mass_g':body_mass_g,
              'sex':sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df=user_input_features()
    


# Combines user input features with entire penguins datasets

penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# Encoding of ordinal features

encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
# First raw of user input
df= df[:1]

# Displays the user input features
st.subheader("Select features")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awating CSV file to be uploaded, currently using example input parameters(shown below) ")
    st.write(df)

# Reads in saved clf model
load_clf =pickle.load(open('D:\Penguin_classification_app\penguin_clf.pkl','rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba=load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species =  np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader("Prediction Probablity")
st.write(prediction_proba)



