# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 08:56:18 2022

@author: alejo
"""

import streamlit as st

st.set_page_config(page_title = "BMI calculator")



st.sidebar.markdown("BMI calculator")
st.sidebar.write(f"hello {st.session_state['user']}")

st.markdown("BMI calculator")
st.write('your BMI is 9999')