#!/usr/bin/python 
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:18:28 2019

@author: TS Liu
"""
from __future__ import print_function

import os
import sys

import numpy as np
from nptdms import TdmsFile
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import pandas as pd

def TdmstoList( tdms_file, consistance, Data_index = None):
    '''
    将tdms文件导出并提取数据作为列表
    consistance用于将放大器输出电压换算为电流
    Data_index = none ，数据截取到指定的索引，不赋值默认全部导入
    '''
    
    channel = tdms_file.object('Data', 'RT Voltage [V]')
    channel2 = tdms_file.object('Data','Sampling Value [V]')
    if Data_index == None:
        RT = channel.data/consistance
        SV = channel2.data
    else:
        RT = channel.data[0:Data_index]/consistance
        SV = channel2.data[0:Data_index]
    return RT, SV                       #注意RT与SV均为ndarray类型，不是真的list


def linearModel(data):
    '''
    data : DataFrame, 建模数据
    '''
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    #加入常量变量
    X = sm.add_constant(data[features])
    #构建模型
    re = trainModel(X, Y)
    modelSummary(re)

    
def ModelSet( X, Y):
    '''
    建立模型并拟合
    '''
    df = pd.DataFrame([ Y, X], index = list('xy')).T
    return df


def visualizeModel(re, data, features, labels):
    """
    模型可视化
    """
    # 计算预测结果的标准差，预测下界，预测上界
    prstd, preLow, preUp = wls_prediction_std(re, alpha=0.05)
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif']=['SimHei']
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 在Matplotlib中显示中文，需要使用unicode
    ax.set_title(u'%s' % "线性回归统计分析示例")
    # 画点图，用蓝色圆点表示原始数据
    ax.scatter(data[features], data[labels], color='b',
            label=u'%s: $y = x + \epsilon$' % "真实值")
    # 画线图，用红色虚线表示95%置信区间
    ax.plot(data[features], preUp, "r--", label=u'%s' % "95%置信区间")
    ax.plot(data[features], re.predict(data[features]), color='r',
        label=u'%s: $y = %.3fx$'\
        % ("预测值", re.params[features]))
    
    ax.plot(data[features], preLow, "r--")
    legend = plt.legend(shadow=True)
    legend.get_frame().set_facecolor('#6F93AE')
    plt.show()
    
    
def trainModel(X, Y):
    '''
    训练模型
    '''
    model = sm.OLS(Y, X)
    re = model.fit()
    return re


def modelSummary(re):
    """
    分析线性回归模型的统计性质
    """
    # 整体统计分析结果
    print(re.summary())
    # 用f test检测x对应的系数a是否显著
    print("检验假设x的系数等于0：")
    print(re.f_test("x=0"))
    # 用f test检测常量b是否显著
    print("检测假设const的系数等于0：")
    print(re.f_test("const=0"))
    # 用f test检测a=1, b=0同时成立的显著性
    print("检测假设x的系数等于1和const的系数等于0同时成立：")
    print(re.f_test(["x=1", "const=0"]))
    

    
if __name__ == '__main__':
    
    '''
    #每组数据单独拟合线性度均非常好，但实际上第一组与后面几组的拟合方程不匹配
    tdms_file = TdmsFile("C:\\Users\\haier\\Desktop\\刘天硕\\放大器拟合数据\\TLC2201 1kHz 5V\\10E3.tdms")
    RT, SV = TdmstoList(tdms_file, 1000)
    
    
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
    
    df = ModelSet(SV, RT)      
    linearModel(df)
    
    
    
    
    
    
    
    