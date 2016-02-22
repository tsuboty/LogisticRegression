# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import predict


def draw_prob(df_data):
	df = validate_prediction(df_data,weight_vector)
	plt.scatter(df_data.x, df_data.y, c=df_data.p, cmap='Blues', alpha=0.6)
	plt.xlim([df_data.x.min() -0.1, df.x.max() +0.1])
	plt.ylim([df_data.y.min() -0.1, df.y.max() +0.1])
	plt.colorbar()

	plt.title('plot colored by probability', size=16)