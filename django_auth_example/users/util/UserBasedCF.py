#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/3/30 9:00
# @Author : MCM
# @Site : 
# @File : UserBasedCF.py
# @Software: PyCharm


import sys
import random
import math
from operator import itemgetter

random.seed(0)
user_sim_mat = {}
matrix = []  # 全局变量
matrix2 = []


class UserBasedCF(object):
    """
    类UserBasedCF
    基于用户的协同过滤算法--推荐算法
    """

    def __init__(self):
        self.trainset = {}  # 训练集
        self.testset = {}  # 测试集
        self.initialset = {}  # 存储要推荐的用户的信息
        self.n_sim_user = 20  # 相似用户数量
        self.n_rec_movie = 10  # 在这里修改推荐电影数量

        self.movie_popular = {}
        self.movie_count = 0  # 总电影数量

    @staticmethod
    def loadfile(filename):
        """
        :param filename:load a file
        :return:a generator
        """
        fp = open(filename, 'r', encoding='UTF-8')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
        fp.close()
        print('加载 %s 成功' % filename, file=sys.stderr)

    def generate_dataset(self, filename2, pivot=1.0):
        """
        加载评分数据，，同时分割为训练集和测试集
        :param filename2:传入预先评好分的文件
        :param pivot:
        :return:
        """
        # trainset_len = 0
        # testset_len = 0

        for line in self.loadfile(filename2):
            user, movie, rating = line.split(',')
            if random.random() < pivot:  # pivot=0.7应该表示训练集：测试集=7：3
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = (rating)  # trainset[user][movie]可以获取用户对电影的评分  都是整数
                # trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = rating
                # testset_len += 1

        # print('成功拆分测试集和训练集', file=sys.stderr)
        # print('训练集数量： %s' % trainset_len, file=sys.stderr)
        # print('测试集数量： %s' % testset_len, file=sys.stderr)

    def calc_user_sim(self):
        movie2users = dict()

        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                    # movie2users内容是‘电影号’：{‘喜欢的用户id’}
                movie2users[movie].add(user)  # 看这个电影的用户id
                # #movieId:{'userId','userId'...}

                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        # print ('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print('总电影数为 = %d' % self.movie_count, file=sys.stderr)

        # count co-rated items between users  计算用户之间共同评分的物品
        usersim_mat = user_sim_mat
        # print ('building user co-rated movies matrilx...', file=sys.stderr)

        for movie, users in movie2users.items():  # 通过.items()遍历movie2users这个字典里的所有键、值
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    usersim_mat.setdefault(u, {})
                    usersim_mat[u].setdefault(v, 0)
                    usersim_mat[u][v] += 1 / math.log(1 + len(users))  # usersim_mat二维矩阵应该存的是用户u和用户v之间共同评分的电影数目
        # print ('build user co-rated movies matrix succ', file=sys.stderr)

        # 计算相似矩阵
        simfactor_count = 10  # 相似矩阵的数量（推荐的）
        PRINT_STEP = 20000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1

    # 排序推荐方法
    def recommend(self, user):
        """ Find K similar users and recommend N movies. """
        matrix.clear()  # 每次都要清空
        K = self.n_sim_user  # 这里等于20
        N = self.n_rec_movie  # 这里等于10
        rank = dict()  # 用户对电影的兴趣度
        watched_movies = self.trainset[user]  # user用户已经看过的电影  只包括训练集里的
        # 这里之后不能是训练集
        # watched_movies = self.initialset[user]
        # 将矩阵排序
        for similar_user, similarity_factor in sorted(
                user_sim_mat[user].items(),
                key=itemgetter(1), reverse=True)[0:K]:  # itemgetter(1)表示对第2个域(相似度)排序   reverse=TRUE表示降序
            for imdbid in self.trainset[similar_user]:  # similar_user是items里面的键，就是所有用户   similarity_factor是值，就是对应的相似度
                if imdbid in watched_movies:
                    continue  # 如果该电影用户已经看过，则跳过
                rank.setdefault(imdbid, 0)  # 没有值就为0
                rank[imdbid] += similarity_factor  # rank[movie]就是各个电影的相似度
                # 这里是把和各个用户的相似度加起来，而各个用户的相似度只是基于看过的公共电影数目除以这两个用户看过的电影数量积
                # print(rank[movie])
        # rank_ = dict()
        # 最终排序
        rank_ = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]  # 类型是list不是字典了
        for key, value in rank_:
            matrix.append(key)  # matrix为存储推荐的imdbId号的数组
        return matrix
