# -*- coding:utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np
import json

from embedded_jubatus import Clustering
from jubatus.clustering.types import WeightedDatum
from jubatus.clustering.types import WeightedIndex
from jubatus.clustering.types import IndexedPoint
from jubatus.common import Datum

from config import Config

import logging.config
import logging

import util
import evaluater

class Blobs:
    def __init__(self, config):
        self.config = config

    def get_name(self):
        return "blobs"

    def get_cluster_stds(self):
        return len(self.cluster_stds)

    def set_num_sample(self, n_samples_per_cluster):
        self.config['n_samples_per_cluster'] = n_samples_per_cluster

    def set_num_cluster(self, num_cluster):
        self.config['cluster'] = num_cluster
        
    def build(self):
        self.n_sample = self.config['n_samples_per_cluster'] * self.config['cluster']
        self.cluster_stds = self.config['cluster_stds']
        return self

    def generate_datasets(self):
        i = 0
        for std in self.cluster_stds:
            self.dataset = datasets.make_blobs(
                n_samples = self.n_sample,
                cluster_std = std,
                centers = self.config['centers'])
            i += 1
            yield i,self.dataset

            
def get_logger():
    conf = Config()
    log_conf = conf.get_config('conf', 'log')
    logging.config.fileConfig(log_conf)

    return logging.getLogger()

def evaluate_performance(methods, test_num):
    """
    性能評価を行う
    blobsのクラスタ数とデータ数を変化させる
    クラスタリングアルゴリズムの時間を測定する
    表形式で表示する
    """
    conf = Config()
    perf_conf = (conf.get_sub_config('performance'))
    logger.debug(perf_conf)
    
    FIG_DIR = conf.get_config('fig', 'performance')
    DATA_DIR = conf.get_config('data', 'performance')

    # データセットを作成する
    dataset_conf = conf.get_sub_config('dataset')
    blobs = Blobs(dataset_conf['blobs'])

    # パラメータ
    # TODO:yuhara 効率のよいパラメータ設定方法．外部設定化がよい？
    n_cluster_list = [2, 5, 10, 20, 50]
    n_samples_per_cluster_list = [100, 500, 1000]

    # 色の設定
    colors = util.get_colors()

    # 測定の実施
    for n_cluster in n_cluster_list:
        for n_samples_per_cluster in n_samples_per_cluster_list:
            
            for client_num, (client_name, client) in enumerate(methods.items()):
                for i in test_num:
                    

    
def evaluate_accuracy(methods, datasets, conf):
    picture_dir = conf.get_config('fig', 'accuracy')

    # 色の設定
    colors = util.get_colors()

    for dataset_num, (dataset_name, dataset) in  enumerate(datasets.items()):
        plot_num = 1
        for num, (X, y) in dataset.generate_datasets():
            print("dataset{0}: X={1} y={2}".format(num, X.shape, y.shape))
            for client_num, (client_name, client) in enumerate(methods.items()):
                # print(X[0:4, :])
                # print(StandardScaler().fit_transform(X)[0:4, :])
                i = 0
                client.clear()
                for row in X:
                    client.push([IndexedPoint(str(i), Datum({'x' : row[0], 'y' : row[1]}))])
                    i += 1

                clusters = client.get_core_members_light()

                print(dataset.get_pattern_num(), len(methods), plot_num)

                plt.subplot(dataset.get_pattern_num(), len(methods) , plot_num)
                
                estimate_cluster = [-1] * len(y)
                for idx,cluster in enumerate(clusters):
                    for weighted_index in cluster:           
                        plt.scatter(X[int(weighted_index.id), 0], X[int(weighted_index.id), 1],
                                            color=colors[idx].tolist(), s=10)
                        estimate_cluster[int(weighted_index.id)] = idx

                logger.debug(estimate_cluster)
                evaluater.evaluate(y, estimate_cluster)

                plot_num += 1

    plt.tight_layout()
    plt.savefig(picture_dir+"/clustering.png")
    
def main():
    conf = Config()

    datasets = dict()
    methods = dict()
    
    # logger.debug('generate dataset')
    dataset_conf = conf.get_sub_config('dataset')
    # dataset_config = json.load(conf.get_config('conf', 'dataset'))
    print(dataset_conf)
    blobs = Blobs(dataset_conf['blobs'])
    blobs.build()

    datasets["blobs"] = blobs

    # jubaclusteringの起動
    kmeans_conf = conf.get_sub_config('jubatus/kmeans')
    # print(clustering_conf)
    kmeans = Clustering(kmeans_conf)
    methods["kmeans"] = kmeans

    # dbscanの設定
    dbscan_conf = conf.get_sub_config('jubatus/dbscan')
    dbscan = Clustering(dbscan_conf)
    methods["dbscan"] = dbscan

#    evaluate_accuracy(methods, datasets, conf)
    evaluate_performance(conf)

if __name__ == '__main__':
    # loggerの起動 グローバルにアクセスできるようにここで定義
    logger = get_logger()
    main()
