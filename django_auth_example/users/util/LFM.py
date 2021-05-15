#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/4/23 9:54
# @Author : MCM
# @Site :
# @File : LFM.py
# @Software: PyCharm
import os
import random
import pickle
import pandas as pd
import numpy as np

# file_path = 'ratings.csv'
file_path = '../../users/static/users_resulttable.csv'
# model_path = 'lfm.model'
model_path = 'lfm_train.model'


class LFM:
    def __init__(self, train_size=None, ratio=1):
        self.ratio = ratio
        self.class_count = 5  # 隐类数量
        self.iter_count = 5  # 迭代次数
        self.lr = 0.02  # 步长
        self.lam = 0.01  # lambda正则化参数
        self._load_data(train_size)
        self._init_model()

    def _load_data(self, train_size):
        """
        加载数据
        :return:
        """
        print('loading data...')
        self.data = pd.read_csv(file_path, usecols=range(3))
        self.data.columns = ['user', 'item', 'rating']  # 将这个dataFrame的列分别设为user、item、rating

        self.train_data = self.data.sample(frac=train_size, random_state=10,
                                           axis=0)  # 在data的第一列中随机选出一个例子，random_state意思是随机种子
        self.test_data = self.data[~self.data.index.isin(self.train_data.index)]  # 存入所有合法数据作为测试集

    def _init_model(self):
        """
        初始化用户跟物品的矩阵模型
        :return:
        """
        print('initializing model...')
        self.user_ids = set(self.train_data['user'].values)  # 取出全部user_id
        self.item_ids = set(self.train_data['item'].values)  # 取出全部item_id
        self.items_dict = {user_id: self._get_pos_neg_item(user_id) for user_id in list(self.user_ids)}  # 划分正负样例

        array_p = np.random.randn(len(self.user_ids), self.class_count)  # 创建user隐类矩阵
        array_q = np.random.randn(len(self.item_ids), self.class_count)  # 创建item隐类矩阵
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))  # 将矩阵还原为DataFrame
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))

    def _get_pos_neg_item(self, user_id):
        """
        用于生成正负样例
        根据用户生成正负样例，正样例标记为1，负样例标记为0
        :param user_id:
        :return:
        """
        positive_item_ids = set(
            self.train_data[self.train_data['user'] == user_id]['item'])  # 取出包含该user_id的item，用户有过的item
        negative_item_ids = self.item_ids ^ positive_item_ids  # python求对称差集，返回两个集合不重复的元素。（类变量item_ids中有数据）

        # negative_item_ids = list(negative_item_ids)[:len(positive_item_ids)]  # 正负样例数量相等
        negative_item_ids = list(
            np.random.choice(list(negative_item_ids), int(self.ratio * len(positive_item_ids))))  # 负样例取多倍

        item_dict = {}
        for item in positive_item_ids:
            item_dict[item] = 1
        for item in negative_item_ids:
            item_dict[item] = 0
        return item_dict

    def _predict(self, user_id, item_id):
        """
        根据用户和物品情况计算用户兴趣度
        :param user_id:
        :param item_id:
        :return:
        """
        p = np.mat(self.p.loc[user_id].values)  # 用户对于目标物品属于哪个类的兴趣查找表
        q = np.mat(self.q.loc[item_id].values).T  # 物品属于哪个类的一个概率
        r = (p * q).sum()
        sigmod = 1.0 / (1 + np.exp(-r))  # sigmod，单位阶跃函数,将兴趣度限定在[0,1]范围内
        return sigmod

    def _loss(self, user_id, item_id, y, step):
        """
        计算损失函数
        :param user_id:
        :param item_id:
        :param y:
        :param step:
        :return:
        """
        e = y - self._predict(user_id, item_id)
        print('Step: {}, user_id: {}, item_id: {}, y: {}, loss: {}'.
              format(step, user_id, item_id, y, e))
        return e

    def _optimize(self, user_id, item_id, e):
        """
        优化函数，使用SGD随机梯度下降，添加L2正则化
        :param user_id:
        :param item_id:
        :param e:
        :return:
        """
        gradient_p = -e * self.q.loc[item_id].values
        l2_p = self.lam * self.p.loc[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.loc[user_id].values
        l2_q = self.lam * self.q.loc[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    def train(self):
        """
        迭代训练
        :return:
        """
        # if os.path.exists(model_path):
        #     print("train model exist...")
        #     self.load()
        # else:
        for step in range(0, self.iter_count):
            for user_id, item_dict in self.items_dict.items():
                item_ids = list(item_dict.keys())  # 取出全部item
                random.shuffle(item_ids)  # 随机打乱item的顺序
                for item_id in item_ids:  # 训练的常规步骤
                    e = self._loss(user_id, item_id, item_dict[item_id], step)
                    self._optimize(user_id, item_id, e)
            self.lr *= 0.9
        self.save()

    def predict(self, user_id, N):
        """
        根据用户推荐topN结果
        :param user_id:
        :param N:
        :return:
        """
        user_item_ids = set(self.train_data[self.train_data['user'] == user_id]['item'])
        other_item_ids = self.item_ids ^ user_item_ids  # 求差集，去除用户已选过的item
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        result_list = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)  # 结果根据评分排序
        if N:
            return result_list[:N]
        else:
            return result_list

    def validate(self):
        """
        计算MAE, RMSE指标值
        :return:
        """
        absError_sum = 0.0  # 绝对误差
        squaredError_sum = 0.0  # 均方误差
        setSum = 0
        # i = 0
        for user_id, item_dict in self.items_dict.items():
            print(user_id)
            predict_result = self.predict(user_id, N=None)
            for _, item_rating in enumerate(predict_result):
                item_id = item_rating[0]
                rating = item_rating[1]
                if item_id in item_dict.keys():
                    absError_sum += abs(item_dict[item_id] - rating)
                    squaredError_sum += (item_dict[item_id] - rating) ** 2
                    setSum += 1
        mae = absError_sum / setSum
        rmse = np.sqrt(squaredError_sum / setSum)
        # print("mae:",mae,"rmse",rmse)

        return mae, rmse

    def evaluate(self):
        """
        计算precision, recall评估值
        :return:
        """
        test_users = set(self.test_data['user'].values)
        hit = 0
        recall_sum = 0
        precision_sum = 0
        precision = 0
        recall = 0
        pred_item = {}
        pred_num = 20
        # i = 0

        real_result = self.items_dict.get(len(test_users))
        real_items = list(real_result.keys())  # 真实的items
        pred_result = self.predict(len(test_users), N=pred_num)
        pred_items = [i[0] for i in pred_result]  # 预测的items
        hit += len([i for i in pred_items if i in real_items])  # 预测正确的items

        # pred_item.append(pred_items)
        print(len(test_users))
        print(pred_items)
        recall_sum += len(real_items)
        precision_sum += len(pred_items)


        # for user in test_users:
        #     # i += 1
        #     # if i % 50 == 0:
        #     #     print('calculating %d users' % i)
        #     real_result = self.items_dict.get(user)
        #     real_items = list(real_result.keys())  # 真实的items
        #     pred_result = self.predict(user, N=pred_num)
        #     pred_items = [i[0] for i in pred_result]  # 预测的items
        #     hit += len([i for i in pred_items if i in real_items])  # 预测正确的items
        #
        #     # pred_item.append(pred_items)
        #     print(user)
        #     print(pred_items)
        #
        #     recall_sum += len(real_items)
        #     precision_sum += len(pred_items)

        if precision_sum > 0 and recall_sum > 0:
            precision = hit / (precision_sum * 1.0)
            recall = hit / (recall_sum * 1.0)

        return precision, recall

    def save(self):
        """
        保存模型至本地
        将self.p对象和self.q对象存储到model_path中

        :return:
        """
        f = open(model_path, 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()
        print('save model successfully...')

    def load(self):
        """
        加载本地模型
        将self.p对象和self.q对象从model_path中读取
        :return:
        """

        f = open(model_path, 'rb')
        self.p, self.q = pickle.load(f)
        # print(type(self.p))
        # print(self.p)
        # print(type(self.q))
        # print(self.p)
        f.close()
        print('load model successfully')


if __name__ == '__main__':

    ratio_range = [1, 2, 3, 4, 5]
    mae_rmse = []
    pre_rec = []

    for i in ratio_range:
        lfm = LFM(train_size=0.8, ratio=i)
        if (os.path.exists(model_path)):
            lfm.load()
        else:
            lfm.train()
            print("train over...")
            mae, rmse = lfm.validate()
            print('MAE:', mae, 'RMSE', rmse)    # rmse:均方根误差 mae:平均绝对误差
            mae_rmse.append((mae, rmse))
        pre, rec = lfm.evaluate()
        print('precision:', pre, 'recall:', rec)    # pre:精确度 rec
        # Precision 就是检索出来的条目中（比如：文档、网页等）有多少是准确的，Recall就是所有准确的条目有多少被检索出来了。
        pre_rec.append((pre, rec))
    print(mae_rmse, pre_rec)

    # train_range = [0.9, 0.8, 0.7, 0.6, 0.5]
    # for i in train_range:
    #     print('数据集:', i)
    #     lfm = LFM(train_size=i)
    #     lfm.train()  # 有本地模型可以不用训练
    #     mae, rmse = lfm.validate()
    #     print('MAE:', mae, 'RMSE', rmse)
    #     pre, rec = lfm.evaluate()
    #     print('precision:', pre, 'recall:', rec)
    #     # res = lfm.predict(1, N=10)
    #     # print(res)
