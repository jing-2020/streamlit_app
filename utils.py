# coding=utf-8
#@author:      ChengJing
#@name:        utils.py
#@datetime:    2021/5/30 22:12
#@software:    PyCharm
#@description:

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
from shapely.geometry import Polygon,MultiPolygon
from scipy.spatial.distance import pdist
import folium
from folium.plugins import DualMap,MousePosition
from branca.colormap import linear
import utm

def area_id_set(file=r".\数据\24分区.xlsx"):
    '''
    根据输入的文件统计每个分区的节点索引集合
    Args:
        file: xlsx file, which save node and node's area id

    Returns:
        area_dict: dict, key is area index and value is a set of nodes belong to this area
    '''
    df = pd.read_excel(file,index_col=None,header=None)
    df[1]=np.arange(0,464)
    area_dict = {}
    for i in df[0].unique():
        area_dict[i] = df[df[0]==i][1].values
    return area_dict

class trans_utm_longlat():
    '''
    将utm类型的坐标转换成经纬度格式的坐标
    '''
    def __init__(self,utms,longlat):
        '''
        Args:
            utms: list,
                  list containing two node's x, y coordinates, like this: [[x,y],[x,y]]
            longlat: list,
                    list containing two node's x, y coordinates in
                    longitude, latitude in the format, like this: [[long,lat],[long,lat]]
        '''
        A = utms
        B = []
        for ll in longlat:
            longitude = ll[0]
            latitude = ll[1]
            utm_coords = utm.from_latlon(latitude, longitude)
            self.zone_number = utm_coords[2]
            self.zone_letter = utm_coords[3]
            B.append(np.array(utm_coords[0:2]))

        # Compute center points
        self.cpA = np.mean(A, axis=0)
        self.cpB = np.mean(B, axis=0)
        # Compute distance to each center point
        distA = np.mean([pdist([A[0], self.cpA])[0], pdist([A[1], self.cpA])[0]])
        distB = np.mean([pdist([B[0], self.cpB])[0], pdist([B[1], self.cpB])[0]])
        # Compute ratio
        self.ratio = distB / distA

    def transform(self,data_pos):
        '''
        Args:
            data_pos: list,ndarray,
                      a data set for transform, like this: [[x,y],[x,y],...,[x,y],[x,y]]
        Returns:
            ndarray,transformed data in longitude, latitude in the format
        '''
        pos = data_pos
        s = (np.array(pos) - self.cpA)*self.ratio + self.cpB
        easting, northing = s[:,0],s[:,1]
        lat, long = utm.to_latlon(easting, northing, self.zone_number, self.zone_letter)
        return (long, lat)


# @st.cache
class trans_utm_plot():

    def __init__(self,file=r'数据/24.geojson',is_add_data=True):
        utm=[(73185.78, 45250.2),(72247.25, 44956.51)]
        longlat=[(120.62,31.32),(120.61,31.31)]
        trans = trans_utm_longlat(utm, longlat)
        df = gpd.read_file(file)
        self.df_ll = gpd.GeoDataFrame()
        df_ll_geo = []
        for num,geo in enumerate(df.geometry):
            coords = np.array(geo[0].exterior.coords)
            lon,lat = trans.transform(coords)
            df_ll_geo.append(MultiPolygon([Polygon(zip(lon,lat))]))
        self.df_ll.geometry = df_ll_geo
        self.df_ll['name'] = df['Burst_id']
        if is_add_data:
            self.add_data()

    def save_geo_file(self,path='t24.geojson'):
        self.df_ll.to_file(path,driver='GeoJSON')


    def add_data(self):
        df=pd.read_excel(r'数据/流量分析.xlsx',sheet_name='分区',header=0,index_col=None)
        name = self.df_ll['name'].values
        area_set = area_id_set()
        data = []
        large=[]
        for i in name:
            large.append(len(area_set[int(i.split('_')[1])]))
            data.append(np.round(df['识别正确率%'][int(i.split('_')[1])-1],2))
        self.df_ll['区域识别率']= data
        self.df_ll['区域节点数'] = large

    @classmethod
    def show_Choropleth(cls,data,color_r):
        df_ll = gpd.read_file('t24.geojson')
        top_data=[]
        color_rs=[]
        for i in df_ll.name:
            top_data.append(data[int(i.split('_')[1])-1])
            color_rs.append(color_r[int(i.split('_')[1])-1])
        datas = pd.DataFrame()
        datas['id'] = df_ll.name
        datas['top'] = color_rs
        type_data=[]
        for i in top_data:
            if i==0:
                type_data.append('No burst')
            else:
                type_data.append('top'+str(int(i))+'分区')
        df_ll['area_type'] = type_data
        map = folium.Map((31.32,120.62),zoom_start=12
                         ,tiles='cartodbpositron'
                         )
        folium.Choropleth(
            df_ll.to_json(),
            name='预测结果',
            data = datas,
            columns=['id','top'],
            key_on='feature.properties.name',
            legend_name='预测可能性',
            fill_color='YlOrRd',
            nan_fill_color='#888888',
            fill_opacity=0.7,
            line_opacity=0.2,
            line_color='blue',
            highlight=True,
            smooth_factor=1.0,
            tooltip=folium.GeoJsonTooltip(fields=['name','area_type'],
                                          aliases=['分区ID：','类别：'],
                                          localize=True,
                                          sticky=False,
                                          labels=True,
                                          style="""
                                                background-color: #F0EFEF;
                                                border: 2px solid black;
                                                border-radius: 3px;
                                                box-shadow: 3px;
                                            """,
                                          max_width=800,
                                          )
        ).add_to(map)
        return map

    def show(self,map=None,colormap=None,text="区域识别率"):
        if map is None:
            self.add_data()
            map = folium.Map((31.32,120.62),zoom_start=12
                         ,tiles='cartodbpositron'
                         )
        if colormap is None:
            colormap = linear.YlGn_09.scale(0, 100)
        folium.GeoJson(self.df_ll.to_json(),
                    name='分区',
                    smooth_factor=1.0,
                    style_function=lambda feature: {
                           "fillColor": colormap(feature['properties'][text]),
                           # "color": "blue",
                           "weight": 0.5,
                           "dashArray": "5, 5",
                           'line_opacity' : 0.1,
                           "fillOpacity": 0.7 },
                    highlight_function=lambda x:{
                                        'weight': 1 + 2,
                                        'fillOpacity': 0.7 + .2},
                    tooltip=folium.GeoJsonTooltip(fields=['name',text,'区域节点数'],
                                                  aliases=['分区ID：',"top1识别率：",'区域节点数：'],
                                                  localize=True,
                                                  sticky=False,
                                                  labels=True,
                                                  style="""
                                                      background-color: #F0EFEF;
                                                      border: 2px solid black;
                                                      border-radius: 3px;
                                                      box-shadow: 3px;
                                                  """,
                                                  max_width=800,
                                                  ),
            ).add_to(map)
        colormap.caption = text
        colormap.add_to(map)

        folium.LayerControl().add_to(map)
        return map

    def show_DualMap(self,text1="区域识别率",text2='区域节点数'):
        map = DualMap((31.32,120.62),zoom_start=12
                         ,tiles='cartodbpositron')
        colormap1 = linear.YlGn_09.scale(0, 100)
        colormap2 = linear.OrRd_09.scale(0, 30)
        self.show(map.m1,colormap1,text1)
        self.show(map.m2,colormap2,text2)
        return map

    def show_2_fig(self,text1="区域识别率",text2='区域节点数'):
        from branca.element import Figure
        fig = Figure()
        fig1 = fig.add_subplot(1,2,1)
        fig2 = fig.add_subplot(1, 2, 2)
        colormap1 = linear.YlGn_09.scale(0, 100)
        colormap2 = linear.OrRd_09.scale(0, 30)
        map1=self.show(colormap=colormap1,text=text1)
        map2=self.show(colormap=colormap2,text=text2)
        fig1.add_child(map1)
        fig2.add_child(map2)
        fig.save('2fig.html')


if __name__ == '__main__':
    print(area_id_set())
    # trans_utm_wgs()
    # df_ll = gpd.read_file('t24.geojson')
    # map = folium.Map((31.32, 120.62), zoom_start=12
    #                  , tiles='cartodbpositron'
    #                  )
    # state_data = pd.DataFrame()
    # state_data['id'] = df_ll.name
    # state_data['gmv'] = np.random.rand(24)
    # state_data['ss'] = np.linspace(1, 100, 24)
    # folium.Choropleth(
    #     geo_data=df_ll,
    #     name='分区',
    #     data=state_data,
    #     columns=['id','ss','gmv'],
    #     key_on='feature.properties.name',
    #     fill_color='YlGn',
    #     nan_fill_color='#0000ff',
    #     fill_opacity=0.7,
    #     line_opacity=0.2,
    #     line_color='blue',
    #     legend_name='分区',
    #     highlight=True,
    #     smooth_factor=1.0,
    #     tooltip=folium.GeoJsonTooltip(fields=['name'])
    # ).add_to(map)
    # map.save('out.html')
