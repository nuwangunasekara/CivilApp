import streamlit as st
import numpy as np
import pandas as pd

import pickle

root_dir = './'
automl = pickle.load(open(root_dir + 'automl_model.pkl', 'rb'))
sample_x = pickle.load(open(root_dir + 'testdata.pkl', 'rb'))

input_container = st.container()
output_container = st.container()
ic1,ic2=input_container.columns(2)

#       fy     t       L     d1     d2    d3   d12   d13
# 6  436.0  1.92  2000.0  359.2  128.0  24.7  7.68  7.68


with ic1:
    st.subheader('Axial-capacity predictor', divider='rainbow')

    dynamic_label_placeholder = st.empty()

    fy=st.number_input("**fy:**",min_value=0.0,max_value=1000.0,step=1.0,value=436.0)
    t=st.number_input("**t:**",min_value=0.0,max_value=1000.0,step=1.0,value=1.92)
    L=st.number_input("**L:**",min_value=0.0,max_value=3000.0,step=1.0,value=2000.0)
    d1=st.number_input("**D1:**", min_value=0.0,max_value=2000.0,step=1.0,value=359.2)
    d2=st.number_input("**D2:**", min_value=0.0,max_value=1000.0,step=1.0,value=128.0)
    d3 = st.number_input("**D3:**", min_value=0.0, max_value=1000.0, step=1.0, value=24.7)
    d12 = st.number_input("**D12:**", min_value=0.0, max_value=1000.0, step=1.0, value=7.68)
    d13 = st.number_input("**D13:**", min_value=0.0, max_value=1000.0, step=1.0, value=7.68)

    x = sample_x.copy()

    x['fy'] = fy
    x['t'] = t
    x['L'] = L
    x['d1'] = d1
    x['d2'] = d2
    x['d3'] = d3
    x['d12'] = d12
    x['d13'] = d13

    pred = automl.model.predict(x)

    dynamic_label_placeholder.write(f"**Axial-capacity for below values** = **{round(pred[0], 1)}**")
