# -*- coding:utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np

from config import Config

def main():
    conf = Config()
    print(conf.get_config('conf', 'dataset'))

if __name__ == '__main__':
    main()
