# -*- coding:utf-8 -*-

from __future__ import print_function
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from collections import defaultdict
import numpy as np
import json
import time
import os


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

    def set_samples_per_cluster_list(self, n_samples_per_cluster_list):
        self.config['n_samples_per_cluster'] = n_samples_per_cluster_list
        return self

    def set_num_cluster_list(self, num_cluster_list):
        self.config['cluster'] = num_cluster_list
        return self

    def set_num_cluster_stds_list(self, num_cluster_stds_list):
        self.config['cluster_stds'] = num_cluster_stds_list
        return self

    def generate(self):
        step = 10
        i = 1
        for num_sample in self.config['n_samples_per_cluster']:
            for num_cluster in self.config['cluster']:
                total_num_sample = num_sample * num_cluster
                centers = [(x,x) for x in range(0, step*num_cluster, step)]
                for cluster_std in self.config['cluster_stds']:
                    print(centers)
                    self.dataset = datasets.make_blobs(
                        n_samples = total_num_sample,
                        cluster_std = cluster_std,
                        centers = centers)
                    yield (i,num_cluster,num_sample,cluster_std,self.dataset)
                    i += 1
            
def get_logger():
    conf = Config()
    log_conf = conf.get_config('conf', 'log')
    logging.config.fileConfig(log_conf)

    return logging.getLogger()

class sklearn_client():
    def __init__(self, method_conf):
        if method_conf['method'] in 'kmeans':
            self.client = KMeans(n_clusters = method_conf['parameter']['k'])
        elif method_conf['method'] in 'gmm':
            self.client = GMM(n_components = method_conf['parameter']['k'])
        elif method_conf['method'] in 'dbscan':
            self.client = DBSCAN(eps = method_conf['parameter']['eps'], min_samples = method_conf['parameter']['min_core_point'])
        self.method_conf = method_conf
        self.revision = 0

    def push(self, data):
        self.client.fit(data)
        self.revision = 1
        if self.method_conf['method'] in 'gmm':
            # gmmの場合はpredictを適用してラベルのみを返却する
            return self.client.predict(data)
        return self.client

    def clear(self):
        pass

    def get_revision(self):
        return self.revision
        
def evaluate_clustering_performance(test_num, is_sklearn=False):
    """
    性能評価を行う
    blobsのクラスタ数とデータ数を変化させる
    クラスタリングアルゴリズムの時間を測定する
    表形式で表示する
    """
    logger.info("===== is_sklearn {0} ======".format(is_sklearn))
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
    # n_cluster_list = [2, 5, 10, 20, 50]
    n_cluster_list = [3]
    # n_samples_per_cluster_list = [10, 100, 500]
    n_samples_per_cluster_list = [10]

    # データセットにパラメータを設定する
    blobs.set_num_cluster_list(n_cluster_list)
    blobs.set_samples_per_cluster_list(n_samples_per_cluster_list)
    blobs.set_num_cluster_stds_list([1.0])

    # 色の設定
    colors = util.get_colors()
    # データの保存場所の設定
    DATA_DIR = conf.get_config('data', 'performance')

    # 測定の実施
    # methods = ['kmeans', 'gmm', 'dbscan']
    methods = ['gmm']
    result = defaultdict(util._factory)
    for data_i, num_cluster, num_sample, cluster_std, (X, y) in blobs.generate():
        print("{0} / {1} start.".format(data_i, len(n_cluster_list)*len(n_samples_per_cluster_list)))
        logger.debug("{0} {1} {2} {3}".format(data_i, num_cluster, num_sample, cluster_std))
        for method in methods:
            logger.debug("method {0}".format(method))
            method_conf = dict()
            if method in 'kmeans':
                method_conf = conf.get_sub_config('jubatus/kmeans')
                method_conf['parameter']['k'] = num_cluster
            elif method in 'gmm':
                method_conf = conf.get_sub_config('jubatus/gmm')
                method_conf['parameter']['k'] = num_cluster
            elif method in 'dbscan':
                method_conf = conf.get_sub_config('jubatus/dbscan')
            # クラスタリング1回のみ行うように調整
            method_conf['compressor_parameter']['bucket_size'] = num_sample * num_cluster
            logger.debug("{0}".format(method_conf))
            clustering_client = Clustering(method_conf)
            if is_sklearn:
                clustering_client = sklearn_client(method_conf)
            duration = 0
            for test_i in range(test_num):
                point_i = 0
                clustering_client.clear()
                indexed_points = []
                if is_sklearn:
                    indexed_points = X
                else:
                    for row in X:
                        indexed_points.append(IndexedPoint(str(point_i), Datum({'x' : row[0], 'y' : row[1]})))
                        point_i += 1
                start_time = time.time()
                model = clustering_client.push(indexed_points)
                end_time = time.time()
                duration += (end_time - start_time)
                logger.debug("duration {0}".format((end_time - start_time)))
                logger.debug("revision {0}".format(clustering_client.get_revision()))
            average_duration = duration / test_num
            logger.info("average {0}".format(average_duration))
            result[method][num_cluster][num_sample] = average_duration

            with open(os.path.join(DATA_DIR, "perfomance.csv"), 'a') as f:
                f.write("{0},{1},{2},{3}\n".format(method, num_cluster, num_sample, average_duration))

        print("  {0} end".format(data_i))


    with open(os.path.join(DATA_DIR, "perfomance.csv"), 'a') as f:
        for method in methods:
            for n_cluster in n_cluster_list:
                for n_sample in n_samples_per_cluster_list:
                    f.write("{0},{1},{2},{3}\n".format(method, n_cluster, n_sample, result[method][n_cluster][n_sample]))
        f.write("\n")
    return result

    
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
    # dataset_conf = conf.get_sub_config('dataset')
    # dataset_config = json.load(conf.get_config('conf', 'dataset'))
    # print(dataset_conf)
    # blobs = Blobs(dataset_conf['blobs'])
    # blobs.build()

    # datasets["blobs"] = blobs

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
    result = evaluate_clustering_performance(5, True)
    print("{0}".format(json.dumps(result, indent=4)))
    logger.info("{0}".format(json.dumps(result, indent=4)))

if __name__ == '__main__':
    # loggerの起動 グローバルにアクセスできるようにここで定義
    logger = get_logger()
    main()
