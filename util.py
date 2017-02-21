# -*- coding:utf-8 -*-

import numpy as np

def get_colors():
    # 色の設定
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    return colors
