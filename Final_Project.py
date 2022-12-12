import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
s = pd.read_csv("social_media_usage.csv", usecols = ['web1h','income','educ2', 'par', 'marital', 'gender', 'age'])
st.write("Hello ,let's learn how to build a streamlit app together")
st.dataframe(s.head())