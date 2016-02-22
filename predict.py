# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from make_data import *

def validate_prediction(df_data, weight_vector):

    a, b, c = weight_vector
    df_data['pred'] = df_data.apply(lambda row : 1 if (a*row.x + b*row.y + c) >0 else 0, axis=1)
    df_data['p'] = df_data.apply(lambda row :  sigmoid(a*row.x + b*row.y + c), axis=1)

    return df_data