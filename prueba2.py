# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 19:01:06 2022

@author: alejo
"""

https://coderzcolumn.com/tutorials/data-science/basic-dashboard-using-streamlit-and-matplotlib

import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

import warnings
warnings.filterwarnings("ignore")

####### Load Dataset #####################

breast_cancer = datasets.load_breast_cancer(as_frame=True)

breast_cancer_df = pd.concat((breast_cancer["data"], breast_cancer["target"]), axis=1)

breast_cancer_df["target"] = [breast_cancer.target_names[val] for val in breast_cancer_df["target"]]

measurements = breast_cancer_df.drop(labels=["target"], axis=1).columns.tolist()
