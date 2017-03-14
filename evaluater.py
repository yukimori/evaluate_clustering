# -*- coding:utf-8 -*-

from sklearn import metrics

def evaluate(expect_cluster, estimate_cluster):
    ari_score = metrics.adjusted_rand_score(expect_cluster, estimate_cluster)
    mibs_score = metrics.adjusted_mutual_info_score(expect_cluster, estimate_cluster)

    print("#ari, mibs")
    print("{0},{1}".format(ari_score, mibs_score))

    
