# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 08:55:09 2022

@author: alejo
"""

import streamlit as st

st.set_page_config(page_title = "Main page")

st.markdown("A bunch of calculators")

user = st.text_input('enter your name')
update = st.button('update user')

if 'user' not in st.session_state:
    st.session_state['user'] = user

if update:
    st.session_state['user'] = user
    
    
numero = st.number_input('enter number')   

if 'number' not in st.session_state:
    st.session_state['number'] = numero

if update:
    st.session_state['number'] = numero

    
st.sidebar.write(f"hello {st.session_state['user']}")

st.write(st.session_state)
