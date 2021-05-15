#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/3/30 9:12
# @Author : MCM
# @Site : 
# @File : ItemBasedCF.py
# @Software: PyCharm
import math
import sys
from operator import itemgetter
from random import random

from users.util.UserBasedCF import matrix2


class ItemBasedCF(object):
    """Top-N推荐系统-基于项目的协同过滤"""

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20
        self.n_rec_movie = 10

        self.movie_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

    @staticmethod
    def loadfile(filename):
        """

        :param filename:加载文件
        :return: return a generator
        """
        fp = open(filename, 'r', encoding='UTF-8')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
        fp.close()
        print('加载 %s 成功' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=1.0):
        """加载额定数据并将其拆分为训练集和测试集"""
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating = line.split(',')
            rating = float(rating)
            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})

                self.trainset[user][movie] = float(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})

                self.testset[user][movie] = float(rating)
                testset_len += 1

        print('训练集 = %s' % trainset_len, file=sys.stderr)
        print('测试集 = %s' % testset_len, file=sys.stderr)

    def calc_movie_sim(self):
        """ 计算电影相似度矩阵 """
        print('正在计算电影数量和受欢迎程度...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for movie in movies:
                # count item popularity
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        self.movie_count = len(self.movie_popular)
        print('电影总数 = %d' % self.movie_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.movie_sim_mat
        # print('building co-rated users matrix...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    itemsim_mat.setdefault(m1, {})
                    itemsim_mat[m1].setdefault(m2, 0)
                    itemsim_mat[m1][m2] += 1 / math.log(1 + len(movies) * 1.0)

        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('正在计算电影相似因素(%d)' %
                          simfactor_count, file=sys.stderr)

    def recommend(self, user):
        """ Find K similar movies and recommend N movies. """
        K = self.n_sim_movie
        N = self.n_rec_movie
        matrix2.clear()
        rank = {}
        watched_movies = self.trainset[user]

        for movie, rating in watched_movies.items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating
        # return the N best movies
        rank_ = sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
        for key, value in rank_:
            matrix2.append(key)  # matrix为存储推荐的imdbId号的数组
        # print(matrix2)
        return matrix2
