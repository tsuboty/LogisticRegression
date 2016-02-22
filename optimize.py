# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from numpy.random import randint
from scipy import optimize
from make_data import *
from estimate_param import *


weight_vector = np.random.rand(3)


def draw_split_line(weight_vector):
    '''分離線を描画する関数
    '''
    a,b,c = weight_vector
    x = np.array(range(-10,10,1))
    y = (a * x + c)/-b
    plt.plot(x,y, alpha=0.3)
    plt.show()

#適当な初期値の重みを設定して、分離面を描画してみる
weight_vector = np.random.rand(3)




#データを作成するして、プロットする
df_data = make_data(1000)

weight_vector = estimate_weight(df_data, weight_vector)
draw_split_line(weight_vector)

plt.show()