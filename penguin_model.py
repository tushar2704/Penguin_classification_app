#Creating model for penguin_clf_app.py

# Importing required libraries 
import pandas as pd


penguins_df = pd.read_csv('penguins_cleaned.csv')
# Ordinal feature encoding
df = penguins_df.copy()
target = 'species'
encode =['sex', 'island']





for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df=pd.concat([df, dummy], axis=1)
    del df[col]


target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)


# Seprating X & y
X= df.drop('species', axis=1)
y= df['species']

# Building random forest model
from sklearn.ensemble import RandomForestClassifier

clf= RandomForestClassifier()
clf.fit(X,y)

# Saving the model
import pickle
pickle.dump(clf, open('penguin_clf.pkl', 'wb'))


