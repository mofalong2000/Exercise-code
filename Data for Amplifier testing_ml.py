#!/usr/bin/python 
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:18:28 2019

@author: TS Liu
"""
import numpy as np
from matplotlib import pyplot as plt
from nptdms import TdmsFile
import pandas as pd
from sklearn import linear_model

def TdmstoList( tdms_file, consistance, Data_index = None):
    '''
    将tdms文件导出并提取数据作为列表
    consistance用于将放大器输出电压换算为电流
    Data_index = none ，数据截取到指定的索引，不赋值默认全部导入
    '''
    
    channel = tdms_file.object('Data', 'RT Voltage [V]')
    channel2 = tdms_file.object('Data','Sampling Value [V]')
    print(type(tdms_file))
    if Data_index == None:
        RT = channel.data/consistance
        SV = channel2.data
    else:
        RT = channel.data[0:Data_index]/consistance
        SV = channel2.data[0:Data_index]
    return RT, SV                       #注意RT与SV均为ndarray类型，不是真的list

def ModelSet( X, Y):
    '''
    建立模型并拟合
    '''
    
    df = pd.DataFrame([ Y, X], index = list('xy')).T
    model = linear_model.LinearRegression()
    features = ["x"]
    labels = ["y"]
    model.fit( df[features], df[labels])
    return model, df

def ModelViewer( model, df):
    ''' 
    模型可视化及参数输出
    '''
    
    a,b = model.coef_, model.intercept_
    print('模型斜率为 %f' % a)
    print('模型截距为 %f' % b)
    features = ["x"]
    labels = ["y"]
    # 均方差(The mean squared error)，均方差越小越好
    error = np.mean((model.predict(df[features]) - df[labels]) ** 2)
    # 决定系数(Coefficient of determination)，决定系数越接近1越好
    score = model.score(df[features], df[labels])
    print('均方差为 %.10f' % error)
    print('决定系数为 %f' % score)
    
    plt.scatter( df[features], df[labels], color='blue')
    plt.plot( df[features], model.predict(df[features]), color='red', linewidth=2)
    plt.title('Data analysing for Amplifier')
    #plt.xlim(-0.001,0.002)
    #plt.ylim(0,0.005)
    plt.xlabel('RT Current [A]')
    plt.ylabel('Sampling Value [V]')
    plt.show()


    
if __name__ == '__main__':

    #每组数据单独拟合线性度均非常好，但实际上第一组与后面几组的拟合方程不匹配
    tdms_file = TdmsFile("C:\\Users\\haier\\Desktop\\刘天硕\\放大器拟合数据\\TLC2201 1kHz 5V\\10E3.tdms")
    RT, SV = TdmstoList( tdms_file, 1000)
    
    '''
    #下面的代码有问题，本打算用于所有数据一起拟合，虽然看起来效果很好，但决定系数很低
    tdms_fileE3 = TdmsFile("C:\\Users\\haier\\Desktop\\刘天硕\\放大器拟合数据\\TLC2201 1kHz 5V\\10E3.tdms")
    tdms_fileE4 = TdmsFile("C:\\Users\\haier\\Desktop\\刘天硕\\放大器拟合数据\\TLC2201 1kHz 5V\\10E4.tdms")
    tdms_fileE5 = TdmsFile("C:\\Users\\haier\\Desktop\\刘天硕\\放大器拟合数据\\TLC2201 1kHz 5V\\10E5.tdms")
    tdms_fileE6 = TdmsFile("C:\\Users\\haier\\Desktop\\刘天硕\\放大器拟合数据\\TLC2201 1kHz 5V\\10E6.tdms")
    tdms_fileE7 = TdmsFile("C:\\Users\\haier\\Desktop\\刘天硕\\放大器拟合数据\\TLC2201 1kHz 5V\\10E7.tdms")
    tdms_fileE8 = TdmsFile("C:\\Users\\haier\\Desktop\\刘天硕\\放大器拟合数据\\TLC2201 1kHz 5V\\10E8.tdms")
    RTE3, SVE3 = TdmstoList( tdms_fileE3, 1000)
    RTE4, SVE4 = TdmstoList( tdms_fileE4, 10000)
    RTE5, SVE5 = TdmstoList( tdms_fileE5, 100000)
    RTE6, SVE6 = TdmstoList( tdms_fileE6, 1000000)
    RTE7, SVE7 = TdmstoList( tdms_fileE7, 10000000)
    RTE8, SVE8 = TdmstoList( tdms_fileE8, 100000000)
             
    RT = np.hstack((RTE3,RTE4))      
    RT = np.hstack((RT,RTE5))    
    RT = np.hstack((RT,RTE6))      
    RT = np.hstack((RT,RTE7))     
    RT = np.hstack((RT,RTE8))                            
    SV = np.hstack((SVE3,SVE4))
    SV = np.hstack((SV,SVE5))
    SV = np.hstack((SV,SVE6))
    SV = np.hstack((SV,SVE7))
    SV = np.hstack((SV,SVE8))               #函数代表数组合并
    '''      
    model, df = ModelSet( RT, SV)
    ModelViewer( model, df)
    
    
    
    
    
    
    
    