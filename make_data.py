# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from numpy.random import randint



def make_data(N, draw_plot=True, is_confused=False, confuse_bin=50):
	'''
	N個のデータセットを生成する関数
	データを複雑にするための機能 is_confusedを実装する
	'''
	np.random.seed(1) #シードを固定して、乱数が毎回同じ出力になるようにする

	feature = np.random.randn(N, 2)  # 標準正規分布による Nx2 の行列
	df = DataFrame(feature, columns=['x','y'])

	# 2分類の付与　g(x,y) = 5x + 3y -1
	df['c'] = df.apply(lambda row : 1 if (5*row.x + 3*row.y -1) > 0 else 0, axis=1)

	# 撹乱:データを少し複雑にするための操作
	if is_confused:
		def get_model_confused(data):
			c = 1 if (data.name % confuse_bin) == 0 else data.c
          		return c

		df['c'] = df.apply(get_model_confused, axis=1)

	# 可視化：どんな感じのデータになったか可視化するモジュール
	# c = df.c つまり2値の0と1で色を分けて表示するようにしてある
	if draw_plot:
     		plt.scatter(x=df.x, y=df.y, c=df.c, alpha=0.6)
        	plt.xlim([df.x.min() -0.1, df.x.max() +0.1])
        	plt.ylim([df.y.min() -0.1, df.y.max() +0.1])
        	plt.show()

	return df

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def get_prob(x,y,weight_vector):
	'''特徴量と重み係数ベクトルを与えると、確率p(c=1 | x,y)を返す関数
		p(x) = w1*x + w2*y + w3
	'''

	feature_vector = np.array([x, y, 1])
	z = np.inner(feature_vector, weight_vector)
	return sigmoid(z)




a = np.array([1,2,3])
b = np.array([2,3,4])

print np.inner(a,b)


