# -*- coding:utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np
import json

from config import Config

import logging.config

class Blobs:
    def __init__(self, config):
        self.config = config
        
    def build(self):
        self.n_sample = self.config['n_samples_per_cluster'] * self.config['cluster']
        self.cluster_stds = self.config['cluster_stds']
        return self

    def generate_datasets(self):
        for std in self.cluster_stds:
            self.dataset = datasets.make_blobs(
                n_samples = self.n_sample,
                cluster_std = std,
                centers = self.config['centers']
                )
            yield self.dataset
    
            
def main():
    conf = Config()

    logging.config.fileConfig(conf.get_config('conf', 'log'))
    logger = logging.getLogger()

    logger.debug('generate dataset')
    dataset_conf = conf.get_sub_config('dataset')
    # dataset_config = json.load(conf.get_config('conf', 'dataset'))
    print(dataset_conf)
    blobs = Blobs(dataset_conf['blobs'])
    blobs.build()

    for X,y in blobs.generate_datasets():
        print(X.shape , " , ", y.shape)
        # print(X[0:4, :])
        # print(StandardScaler().fit_transform(X)[0:4, :])

    anomaly_conf = conf.get_sub_config('anomaly')
    print(anomaly_conf)
        

if __name__ == '__main__':
    main()
