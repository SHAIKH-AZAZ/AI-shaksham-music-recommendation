# main.py
import streamlit as st
import pandas as pd
import joblib
from googleapiclient.discovery import build

# Load the model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
