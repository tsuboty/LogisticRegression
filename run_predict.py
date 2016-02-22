# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from numpy.random import randint
from scipy import optimize

from make_data import *
from optimize import *
from estimate_param import *
from show_p import *
from predict import *

def draw_prob(df_data):
	df = validate_prediction(df_data,weight_vector)
	plt.scatter(df_data.x, df_data.y, c=df_data.p, cmap='Blues', alpha=0.6)
	plt.xlim([df_data.x.min() -0.1, df.x.max() +0.1])
	plt.ylim([df_data.y.min() -0.1, df.y.max() +0.1])
	plt.colorbar()

	plt.title('plot colored by probability', size=16)


plt.figure(figsize=(16, 4))
plt.subplot(1,2,1) #2つの図を並べて表示する準備

#データを作成するして、プロットする
df_data = make_data(1000)

#適当な初期値の重みを設定して、分離面を描画してみる
weight_vector = np.random.rand(3)
draw_split_line(weight_vector)

#最尤推定で重みを推定し、分離面を描画してみる
weight_vector = estimate_weight(df_data, weight_vector)
draw_split_line(weight_vector)
plt.title('plot with split line before/after optimization', size=16)

#pの可視化
plt.subplot(1,2,2)
draw_prob(df_data)