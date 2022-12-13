import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

s = pd.read_csv("social_media_usage.csv", usecols = ['web1h','income','educ2', 'par', 'marital', 'gender', 'age'])

def clean_sm(x):
  return np.where (x==1,1,0)

ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where((s["income"] >0) | (s["income"] <9), s["income"], 99),
    "education":np.where((s["educ2"] >0) | (s["educ2"] <8), s["educ2"], 99),                  
    "age":np.where (s["age"] <98, s["age"], 98),
    "parent":np.where(s["par"] ==1, True, False),
    "married":np.where(s["marital"] ==1, True, False),
    "Female":np.where(s["gender"] ==2, True, False)
})

ss = ss.dropna()

y = ss.iloc[:, 0]

X = ss.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)


st.write("Final Project   -- Author: Kevin Combs")




st.write('Select from the following to calculate the probability of a person will be a LinkedIn user, Press Submit to complete')


col1 = st.columns(1)
form1 = st.form(key='my-form')



myincome = form1.selectbox('Select Income', ['Under 10K', 'Under 20K', 'Under 30K','Under 40K', 'Under 50K', 'Under 75K','Under 100K', 'Under 150K', 'Over 150K',], key=1)
myeduc= form1.selectbox('Select Education Level', ['Less than high school', 'High school incomplete', 'High school graduate','Some college, no degree', 
'Two-year associate degree', 'university degree/Bachelors degree','Some postgraduate or professional', 'Postgraduate or professional degree'], key=2)
myslider = form1.slider(label='Select age', min_value=0, max_value=100, key=3)
mygender = form1.radio(label='Select gender', options=('Male', 'Female'), key=4)
mymar = form1.selectbox('Select Marital Status', ['Married', 'Living with a partner', 'Divorced','Separated', 'Widowed', 'Never been married'], key=5)
myparent = form1.radio(label='Parent', options=('Yes', 'No'), key=6)
submit = form1.form_submit_button('Submit')    

    
# income level 
income=0
if myincome == "Under 10K":
    income = 1
elif myincome == "Under 20K":
    income = 2
elif myincome == "Under 30K":
    income = 3
elif myincome == "Under 40K":
    income = 4
elif myincome == "Under 50K ":
    income = 5
elif myincome == "Under 75K":
    income = 6
elif myincome == "Under 100K":
    income = 7
elif myincome == "Under 150K":
    income = 8
elif myincome == "Over 150K":
    income = 9
else:
    income = 99

#age of person
if myslider <= 97:
    age=myslider
else: 
    age=98

#Education Level
Education=0
if myeduc == "Less than high school":
    educ = 1
elif myeduc == "High school incomplete":
    educ = 2
elif myeduc == "High school graduate":
    educ = 3
elif myeduc == "Some college, no degree":
    educ = 4
elif myeduc == "Two-year associate degree":
    educ = 5
elif myeduc == "university degree/Bachelors degree":
    educ = 6
elif myeduc == "Some postgraduate or professional":
    educ = 7
elif myeduc== "Postgraduate or professional degree":
    educ = 8
else:
    educ = 99

#Gender of Person 
if mygender == "Female":
    gender=1
else: 
    gender = 0

# Marital Status
if mymar == "Married":
     mar_label = 1
elif mymar == "Living with a partner":
     mar_label = 2
elif myeduc == "Divorced":
     mar_label = 3
elif myeduc == "Separated":
     mar_label = 4
elif myeduc == "Widowed":
     mar_label = 5
elif myeduc == "Never been married":
     mar_label= 6
else:
    mar_label = 99

# Parents
if myparent == "Yes":
    parent = 1
else:
    parent = 0




      
if submit:
    newdata = pd.DataFrame({
        "income": [income],
        "education": [educ],
        "age": [age],
        "parent": [parent],
        "married": [mar_label],
        "Female": [gender]
    })

    predictedclass = lr.predict(newdata)
    probs = lr.predict_proba(newdata)
    st.write(f'Debug {income} {age} {educ} {gender} {mar_label} {parent}')
    predict = "1-LinkedIn User" if predictedclass[0] ==1 else "0 - Not LinkedIn User"
    st.write(f"Predicted class: {predict}") 
    st.write(f"Probability that this person using Linkedin:", "{:.4f}".format(round(probs[0][1],4)))
