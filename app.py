import numpy as np
import pandas as pd
import time
import streamlit as st
from streamlit.components.v1 import html
from streamlit_folium import folium_static
from streamlit_echarts import st_pyecharts
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder
import sys
import wntr
import pyecharts.options as opts
from pyecharts.charts import Parallel
import os
path = os.getcwd()

from torch import dtype
sys.path.append(path)
from model_with_se import *
from utils import *

@st.cache
def load_data():
    inp=r"./数据/9月4日高日.inp"
    wn = wntr.network.WaterNetworkModel(inp)
    df = pd.read_excel(r'./数据/流量分析.xlsx',sheet_name='总表',header=0,index_col=None)
    df.columns = ['节点索引','可识别最小流量','可识别最大流量',
                  '无法识别最小流量','无法识别最大流量',
                  '最小爆管流量', '最大爆管流量','属性','所属分区']
    df['节点索引'] = wn.node_name_list[:464]
    return df

@st.cache
def runs(df):
    import torch
    model = torch.load(r'./model.pkl').to('cpu')
    model.eval()
    data = torch.from_numpy(df.values.astype(np.float32))
    py = model(data)
    pre_id = torch.argsort(py, dim=1, descending=True)
    return pre_id

def page1(model,view):
    if model == 'FA-DenseNet':
        st.title('FA-DenseNet模型结果分析')
        if view == "管网可视化":
            with st.beta_container():
                st.header('1.爆管识别分布图')
                with open(r'./plotly_network.html','r') as f:
                    plotlys = f.read()
                html(plotlys,width=700,height=700)
                with open(r'./leaflet_network.html', 'r',encoding='utf-8') as f:
                    leaflet = f.read()
                html(leaflet, width=700, height=700)
                st.info('''
                注释，节点属性值表示识别情况：\n
                属性值-3：表示该节点是监测点；属性值-2：表示该节点可完全识别；\n
                属性值-1：表示该节点可部分识别；属性值-0：表示该节点无法识别。
                ''')

            with st.beta_container():
                st.header('2.管网分区分布图')
                with open(r'./area.html', 'r') as f:
                    plotlys = f.read()
                html(plotlys, width=600, height=700)
                folium_static(trans_utm_plot().show())
        elif view=="数据统计分析":
            data = load_data()
            with st.beta_container():
                st.markdown('''
                ## 识别统计：\n
                合计**464**个节点，共有**368**个节点（占比：**79.3%**）识别正确，
                **58**个节点（占比：**12.5%**）部分正确，**38**个节点（占比：**8.2%**）无法识别。
                ''')
                st.header('数据表')
                gb = GridOptionsBuilder.from_dataframe(data)
                gb.configure_selection('single',rowMultiSelectWithClick=True)
                go = gb.build()
                sel = AgGrid(data,gridOptions=go,
                       fit_columns_on_grid_load=True,
                       update_mode=GridUpdateMode.SELECTION_CHANGED)
                if sel["selected_rows"]:
                    c1,c2 = st.beta_columns([2,1])
                    with c2:
                        st.json(sel["selected_rows"][0])
                    with c1:
                        data=[]
                        data.append([sel["selected_rows"][0]["最小爆管流量"],
                                     sel["selected_rows"][0]['最大爆管流量'],
                                     "总体\n数据集"
                                     ])
                        data.append([sel["selected_rows"][0]['可识别最小流量'],
                                     sel["selected_rows"][0]['可识别最大流量'],
                                     "可识别"
                                     ])
                        data.append([sel["selected_rows"][0]['无法识别最小流量'],
                                     sel["selected_rows"][0]['无法识别最大流量'],
                                     "无法识别"
                                     ])

                        parallel_axis = [
                            {"dim": 0, "name": "流量\n最小值"},
                            {"dim": 1, "name": "流量\n最大值"},
                            {
                                "dim": 3,
                                "name": "类型",
                                "type": "category",
                                "data": ["无法识别","可识别","总体\n数据集"],
                            }]
                        c=(Parallel(init_opts=opts.InitOpts(width="1400px", height="800px"))
                            .add_schema(schema=parallel_axis)
                            .add(
                                series_name="",
                                data=data,
                                linestyle_opts=opts.LineStyleOpts(width=4, opacity=0.5),
                            ))
                        st_pyecharts(c)
                st.info('''
                注释，节点属性值表示识别情况：\n
                属性值-3：表示该节点是监测点；属性值-2：表示该节点可完全识别；\n
                属性值-1：表示该节点可部分识别；属性值-0：表示该节点无法识别。 \n
                ''')
    elif model == 'ELM':
        st.warning('ELM模型还没有分析结果！')

def page2(pre_model):
    if pre_model == 'FA-DenseNet':
        st.title('FA-DenseNet模型分析')
        st.subheader('请选择输入数据的方式:')
        radio = st.radio('',['上传数据文件'])
        if radio == '上传数据文件':
            with st.beta_container():
                file=st.file_uploader('请输入需要预测的压力监测点压降数据：',type='csv')
                if file:
                    st.text('展示数据：')
                    df=pd.read_csv(file,index_col=0)
                    st.dataframe(df)
                    st.info('请检查输入的数据是否正确，不正确请重新输入！')
        else:
            with st.beta_container():
                input_dataframe = pd.DataFrame(
                    '',
                    index=range(10),
                    columns=['监测点%s'%i for i in range(1,25)]
                )
                response = AgGrid(
                    input_dataframe,height=300,
                    editable=True,sortable=False,
                    filter=False,resizable=False,
                    defaultWidth=5,fit_columns_on_grid_load=True,
                    update_mode=GridUpdateMode.MANUAL,
                    data_return_mode=1,key='input_frame')
                if sum(response['data'].any(axis=1)):
                    st.subheader('展示输入的数据:')
                    st.dataframe(response['data'])
                    st.info('请检查输入的数据是否正确，不正确请重新输入！')
                    res_data = []
                    for i in range(10):
                        if response['data'].iloc[i,:].any():
                            res_data.append(pd.to_numeric(response['data'].iloc[i,:]))
                    df = pd.DataFrame(res_data,dtype=np.float32)
                    print(df.values)
        run = st.radio('',['输入数据···','运行模型'])
        if run == '运行模型':
            pre_id = runs(df)
            df['pre_top1_id'] = pre_id.numpy()[:,0]+1
            df['pre_top2_id'] = pre_id.numpy()[:, 1]+1
            df['pre_top3_id'] = pre_id.numpy()[:, 2]+1
            AgGrid(df)
            s=st.selectbox('请选择需要展示的数据索引',[i+1 for i in range(df.shape[0])])
            pre_data = np.zeros(24)
            color_r = np.zeros(24)
            pre_data[df['pre_top1_id'][int(s)]-1] = 1
            color_r[df['pre_top1_id'][int(s)] - 1] = 3
            pre_data[df['pre_top2_id'][int(s)]-1] = 2
            color_r[df['pre_top2_id'][int(s)] - 1] = 2
            pre_data[df['pre_top3_id'][int(s)]-1] = 3
            color_r[df['pre_top3_id'][int(s)] - 1] = 1
            map = trans_utm_plot.show_Choropleth(pre_data,color_r)
            folium_static(map)

    elif pre_model == 'ELM':
        st.warning('ELM模型还没有分析结果！')


st.sidebar.title('管网爆管分析')
sel=st.sidebar.selectbox('++分析选择++',['模型预测','流量分析'])

if sel == '流量分析':
    with st.sidebar.beta_container():
        st.subheader('流量分析：')
        model=st.radio('**选择数学模型**',options=['FA-DenseNet','ELM'])
        view= st.selectbox(
            "**切换可视化界面**",
            ("管网可视化", "数据统计分析")
        )
    with st.spinner('正在使劲跑...'):
        page1(model,view)
elif sel == '模型预测':
    with st.sidebar.beta_container():
        st.subheader('模型预测：')
        pre_model= st.selectbox(
            "**选择数学模型**",
            ('FA-DenseNet','ELM')
        )
    with st.spinner('正在使劲跑...'):
        page2(pre_model)
