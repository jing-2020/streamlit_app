import wntr
import numpy as np
import pandas as pd
from utils import area_id_set

def plot_moniter(wn,num):

    moniters = np.array([447, 314, 276, 327, 153, 111, 230, 0,
                             422, 79, 351, 75, 288, 410, 355, 365,
                             396, 160, 131, 456, 260, 34, 209, 55])
    nodes_atr = np.zeros(wn.num_nodes)
    nodes_atr[moniters] = num
    nodes_attribute_moniter = dict(zip(wn.node_name_list, nodes_atr))

    return nodes_attribute_moniter

def plot_area(wn):
    area_dict = area_id_set()
    nodes_atr = np.zeros(wn.num_nodes)
    for i in range(1,25):
        nodes_atr[area_dict[i]] = i
    nodes = dict(zip(wn.node_name_list, nodes_atr))
    wntr.graphics.plot_interactive_network(
        wn,node_attribute=nodes,node_attribute_name='分区ID',
        node_range=[0,24],
        figsize=[550, 700],
        filename='area.html'
    )


def plot_result(wn):
    excel_file = pd.ExcelFile(r'数据/流量分析.xlsx')
    df_err = excel_file.parse(sheet_name='全部错误',header=0,index_col=None)
    df_rig = excel_file.parse(sheet_name='全部正确',header=0,index_col=None)
    df_part = excel_file.parse(sheet_name='部分正确',header=0,index_col=None)
    nodes_data = np.zeros((wn.num_nodes,5))
    nodes_data[:464,2:5] = excel_file.parse(sheet_name='总表',header=0,index_col=None).iloc[:,[5,6,8]].values
    moniters = np.array([447, 314, 276, 327, 153, 111, 230, 0,
                         422, 79, 351, 75, 288, 410, 355, 365,
                         396, 160, 131, 456, 260, 34, 209, 55])
    nodes_err = df_err.iloc[:,0].values-1
    nodes_right = df_rig.iloc[:,0].values-1
    nodes_part = df_part.iloc[:,0].values-1
    nodes_atr = np.zeros(wn.num_nodes)
    nodes_atr[nodes_right] = 2
    nodes_atr[nodes_part] = 1
    nodes_atr[nodes_err] = 0
    nodes_atr[moniters] = 3
    nodes_attribute = dict(zip(wn.node_name_list, nodes_atr))
    nodes_data[nodes_right,0:2] = df_rig.iloc[:,1:3].values
    nodes_data[nodes_part,0:2] = df_part.iloc[:,1:3].values
    nodes_data[nodes_data == 0] = np.nan
    nodes_data = pd.DataFrame(nodes_data,
                              columns=['可识别最小流量','可识别最大流量','最小爆管流量','最大爆管流量','所属分区'],
                              index=pd.Index(wn.node_name_list))
    wntr.graphics.plot_interactive_network(wn,
                                           node_attribute=nodes_attribute,
                                           node_attribute_name='属性',
                                           node_range=[0,3],
                                           # title='管网爆管节点识别情况',
                                           node_cmap=['#888888','#F28D00','#03F46E','red'],
                                           add_to_node_popup=nodes_data,
                                           figsize=[550, 700]
                                           )


def plot_path(wn,s,t):

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    pipes = results.link['flowrate'].iloc[0, :][results.link['flowrate'].iloc[0, :] < 0].keys()
    g = wn.get_graph()
    pos = nx.get_node_attributes(g, 'pos')
    gg = wn.get_graph()
    for edge in gg.edges:
        for pipe in pipes:
            if pipe in edge:
                g.remove_edge(edge[0], edge[1])
                g.add_edge(edge[1], edge[0],label=edge[2])

    pipes_atr = np.zeros(wn.num_pipes)
    pipes_attribute = dict(zip(wn.pipe_name_list, pipes_atr))
    if nx.has_path(g,s,t):
        path = nx.dijkstra_path(g, s, t)
        index=[]
        for u, v in zip(path, path[1:]):
            for eu,ev,i in gg.edges:
                if set((u,v)) == set((eu, ev)):
                    index.append(i)

        for i in index:
            pipes_attribute[i] = 1

    return pipes_attribute

def plot_reflux(wn):
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    pipes = results.link['flowrate']
    pipes[pipes>0] = 1
    pipes[pipes<0] = -1
    pipes_data = (25-pipes.sum().abs())/2
    pipes_atr = np.zeros(pipes_data.shape[0])
    pipes_atr[pipes_data>0]=1
    pipes_attribution = dict(zip(wn.pipe_name_list,pipes_atr))
    pipes_data = pd.DataFrame(pipes_data,columns=['反流次数'],index=pd.Index(wn.pipe_name_list))
    excel_file = pd.ExcelFile(r'数据/流量分析.xlsx')
    df_err = excel_file.parse(sheet_name='全部错误',header=0,index_col=None)
    df_rig = excel_file.parse(sheet_name='全部正确',header=0,index_col=None)
    df_part = excel_file.parse(sheet_name='部分正确',header=0,index_col=None)
    nodes_data = np.zeros((wn.num_nodes,5))
    nodes_data[:464,2:5] = excel_file.parse(sheet_name='总表',header=0,index_col=None).iloc[:,[5,6,8]].values
    moniters = np.array([447, 314, 276, 327, 153, 111, 230, 0,
                         422, 79, 351, 75, 288, 410, 355, 365,
                         396, 160, 131, 456, 260, 34, 209, 55])
    nodes_err = df_err.iloc[:,0].values-1
    nodes_right = df_rig.iloc[:,0].values-1
    nodes_part = df_part.iloc[:,0].values-1
    nodes_atr = np.zeros(wn.num_nodes)
    nodes_atr[nodes_right] = 2
    nodes_atr[nodes_part] = 1
    nodes_atr[nodes_err] = 0
    nodes_atr[moniters] = 3
    nodes_attribute = dict(zip(wn.node_name_list, nodes_atr))
    nodes_data[nodes_right,0:2] = df_rig.iloc[:,1:3].values
    nodes_data[nodes_part,0:2] = df_part.iloc[:,1:3].values
    nodes_data[nodes_data == 0] = np.nan
    nodes_data = pd.DataFrame(nodes_data,
                              columns=['可识别最小流量','可识别最大流量','最小爆管流量','最大爆管流量','所属分区'],
                              index=pd.Index(wn.node_name_list))
    longlat_map = {'48':(120.62,31.32), '11': (120.61,31.31)}
    wn2 = wntr.morph.convert_node_coordinates_to_longlat(wn, longlat_map)
    wntr.graphics.plot_leaflet_network(wn2,node_attribute=nodes_attribute,
                                       node_attribute_name='属性',
                                       node_range=[0,3],
                                       node_size=5,
                                       add_legend=True,
                                       add_to_node_popup=nodes_data,
                                       link_attribute=pipes_attribution,
                                       link_attribute_name='是否反流',
                                       add_to_link_popup=pipes_data,
                                       link_width=4,
                                       link_range=[0,1]
                                       )



if __name__=='__main__':

    inp=r".\数据\9月4日高日.inp"
    wn = wntr.network.WaterNetworkModel(inp)
    plot_reflux(wn)
    # plot_area(wn)
    # plot_result(wn)
    # moniter = plot_moniter(wn,1)
    # wntr.graphics.plot_interactive_network(wn,
    #                                        # node_attribute=moniter,
    #                                        #  node_attribute_name='is_monitor',
    #                                         add_colorbar=False,figsize=[550,700])

    # longlat_map = {'48':(120.62,31.32), '11': (120.61,31.31)}
    # print(wn.get_node('48').coordinates)
    # print(wn.get_node('11').coordinates)
    # wn2 = wntr.morph.convert_node_coordinates_to_longlat(wn, longlat_map)
    # wntr.graphics.plot_leaflet_network(wn2,node_attribute=moniter,
    #                                     node_attribute_name='is_monitor',
    #                                     node_range=[0,1],
    #                                     node_size=5,link_width=3,
    #                                     add_legend=True)

