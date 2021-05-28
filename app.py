import numpy as np
import pandas as pd
import time
import streamlit as st
from streamlit.components.v1 import html



with open(r'\\psteam\home\Drive\同步文件\第二篇小论文\流量识别\plotly_network.html','r') as f:
    plotlys=f.read()
st.header('1.监测点分布图')
html(plotlys,width=550,height=700)

st.header('2.流量识别分析')
with open(r'\\psteam\home\Drive\同步文件\第二篇小论文\流量识别\leaflet_network.html','r') as f:
    leaflet=f.read()
html(leaflet,width=800,height=500)
