
# coding: utf-8

# In[1040]:

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


# In[1041]:

print('初始化变量...')
names = ['user_id', 'item_id', 'rating', 'timestamp']
trainingset_file = 'dataset/ml-100k/u3.base'
testset_file= 'dataset/ml-100k/u3.test'
n_users = 943
n_items = 1682
ratings = np.zeros((n_users, n_items))


# In[1042]:

df = pd.read_csv(trainingset_file, sep='\t', names=names)
print('载入训练集...')
print('数据集样例为:')
print(df.head())
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
print('载入完成.')
print('打分矩阵规模为 %d*%d.' % (n_users, n_items))
print('测试集有效打分个数为 %d.' % len(df))


# In[1043]:

# 计算矩阵密度
def cal_sparsity():
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print('测试集矩阵密度为: {:4.2f}%'.format(sparsity))

cal_sparsity()
print()


# In[1044]:

def rmse(pred, actual):
    '''计算预测结果的rmse'''
    from sklearn.metrics import mean_squared_error
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual))


# In[1045]:

print('------ Naive算法(baseline) ------')


# In[1046]:

def cal_mean():
    '''Calculate mean value'''
    print('计算总体均值，各user打分均值，各item打分均值...')
    global all_mean, user_mean, item_mean
    all_mean = np.mean(ratings[ratings!=0])
    user_mean = sum(ratings.T) / sum((ratings!=0).T)
    item_mean = sum(ratings) / sum((ratings!=0))
    print('是否存在User/Item 均值为NaN?', np.isnan(user_mean).any(), np.isnan(item_mean).any())
    print('对NaN填充总体均值...')
    user_mean = np.where(np.isnan(user_mean), all_mean, user_mean)
    item_mean = np.where(np.isnan(item_mean), all_mean, item_mean)
    print('是否存在User/Item 均值为NaN?', np.isnan(user_mean).any(), np.isnan(item_mean).any())
    print('均值计算完成，总体打分均值为 %.4f' % all_mean)


# In[1047]:

print('计算训练集各项统计数据...')
cal_mean()


# In[1048]:

def predict_naive(user, item):
    prediction = item_mean[item] + user_mean[user] - all_mean
    return prediction


# In[1049]:

print('载入测试集...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
test_df.head()
predictions = []
targets = []
print('测试集大小为 %d' % len(test_df))
print('采用Naive算法进行预测...')
for row in test_df.itertuples():
    user, item, actual = row[1]-1, row[2]-1, row[3]
    predictions.append(predict_naive(user, item))
    targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[1050]:

print('------ item-item协同过滤算法(相似度未归一化) ------')


# In[1051]:

def cal_similarity(ratings, kind, epsilon=1e-9):
    '''利用Cosine距离计算相似度'''
    '''epsilon: 防止Divide-by-zero错误，进行矫正'''
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


# In[1052]:

print('计算相似度矩阵...')
user_similarity = cal_similarity(ratings, kind='user')
item_similarity = cal_similarity(ratings, kind='item')
print('计算完成.')


# In[1053]:

def predict_itemCF(user, item, k=100):
    '''item-item协同过滤算法,预测rating'''
    nzero = ratings[user].nonzero()[0]
    prediction = ratings[user, nzero].dot(item_similarity[item, nzero])                / sum(item_similarity[item, nzero])
    return prediction


# In[1054]:

print('载入测试集...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
test_df.head()
predictions = []
targets = []
print('测试集大小为 %d' % len(test_df))
print('采用item-item协同过滤算法进行预测...')
for row in test_df.itertuples():
    user, item, actual = row[1]-1, row[2]-1, row[3]
    predictions.append(predict_itemCF(user, item))
    targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[1055]:

print('------ 结合baseline的item-item协同过滤算法(相似度未归一化) ------')


# In[1056]:

def predict_itemCF_baseline(user, item, k=100):
    '''结合baseline的item-item CF算法,预测rating'''
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    prediction = (ratings[user, nzero] - baseline[nzero]).dot(item_similarity[item, nzero])                / sum(item_similarity[item, nzero]) + baseline[item]
    return prediction 


# In[1057]:

print('载入测试集...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
test_df.head()
predictions = []
targets = []
print('测试集大小为 %d' % len(test_df))
print('采用结合baseline的item-item协同过滤算法进行预测...')
for row in test_df.itertuples():
    user, item, actual = row[1]-1, row[2]-1, row[3]
    predictions.append(predict_itemCF_baseline(user, item))
    targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[1058]:

print('------ user-user协同过滤算法(相似度未归一化) ------')

def predict_userCF(user, item, k=100):
    '''user-user协同过滤算法,预测rating'''
    nzero = ratings[:,item].nonzero()[0]
    baseline = user_mean + item_mean[item] - all_mean
    prediction = ratings[nzero, item].dot(user_similarity[user, nzero])                / sum(user_similarity[user, nzero])
    # 冷启动问题: 该item暂时没有评分
    if np.isnan(prediction):
        prediction = baseline[user]
    return prediction

print('载入测试集...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
test_df.head()
predictions = []
targets = []
print('测试集大小为 %d' % len(test_df))
print('采用user-user协同过滤算法进行预测...')

for row in test_df.itertuples():
    user, item, actual = row[1]-1, row[2]-1, row[3]
    predictions.append(predict_userCF(user, item))
    targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[1059]:

print('------ 结合baseline的user-user协同过滤算法(相似度未归一化) ------')

def predict_userCF_baseline(user, item, k=100):
    '''结合baseline的user-user协同过滤算法,预测rating'''
    nzero = ratings[:,item].nonzero()[0]
    baseline = user_mean + item_mean[item] - all_mean
    prediction = (ratings[nzero, item] - baseline[nzero]).dot(user_similarity[user, nzero])                / sum(user_similarity[user, nzero]) + baseline[user]
    if np.isnan(prediction):
        prediction = baseline[user]
    return prediction

print('载入测试集...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
test_df.head()
predictions = []
targets = []
print('测试集大小为 %d' % len(test_df))
print('采用结合baseline的user-user协同过滤算法进行预测...')

for row in test_df.itertuples():
    user, item, actual = row[1]-1, row[2]-1, row[3]
    predictions.append(predict_userCF_baseline(user, item))
    targets.append(actual)
    
print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[1060]:

print('------ 经过修正后的协同过滤 ------')
def predict_biasCF(user, item, k=100):
    '''结合baseline的item-item CF算法,预测rating'''
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    prediction = (ratings[user, nzero] - baseline[nzero]).dot(item_similarity[item, nzero])                / sum(item_similarity[item, nzero]) + baseline[item]
    if prediction > 5:
        prediction = 5
    if prediction < 1:
        prediciton = 1
    return prediction

print('载入测试集...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
test_df.head()
predictions = []
targets = []
print('测试集大小为 %d' % len(test_df))
print('采用结合baseline的item-item协同过滤算法进行预测...')
for row in test_df.itertuples():
    user, item, actual = row[1]-1, row[2]-1, row[3]
    predictions.append(predict_biasCF(user, item))
    targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[1061]:

print('------ Top-k协同过滤(item-item, baseline, 矫正)------')
def predict_topkCF(user, item, k=10):
    '''top-k CF算法,以item-item协同过滤为基础，结合baseline,预测rating'''
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    choice = nzero[item_similarity[item, nzero].argsort()[::-1][:k]]
    prediction = (ratings[user, choice] - baseline[choice]).dot(item_similarity[item, choice])                / sum(item_similarity[item, choice]) + baseline[item]
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction 

print('载入测试集...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
test_df.head()
predictions = []
targets = []
print('测试集大小为 %d' % len(test_df))
print('采用top K协同过滤算法进行预测...')
k = 20
print('选取的K值为%d.' % k)
for row in test_df.itertuples():
    user, item, actual = row[1]-1, row[2]-1, row[3]
    predictions.append(predict_topkCF(user, item, k))
    targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[1062]:

print('经检验，在100k数据上，K=20为佳.')


# In[1063]:

print('------ baseline + item-item + 矫正 + TopK + 归一化矩阵 ------')


# In[1064]:

def cal_similarity_norm(ratings, kind, epsilon=1e-9):
    '''采用归一化的指标:Pearson correlation coefficient'''
    if kind == 'user':
        # 对同一个user的打分归一化
        rating_user_diff = ratings.copy()
        for i in range(ratings.shape[0]):
            nzero = ratings[i].nonzero()
            rating_user_diff[i][nzero] = ratings[i][nzero] - user_mean[i]
#         print(np.sum(rating_user_diff, axis=1)[:20])
        sim = rating_user_diff.dot(rating_user_diff.T) + epsilon
    elif kind == 'item':
        # 对同一个item的打分归一化
        rating_item_diff = ratings.copy()
        for j in range(ratings.shape[1]):
            nzero = ratings[:,j].nonzero()
            rating_item_diff[:,j][nzero] = ratings[:,j][nzero] - item_mean[j]
#         print(np.sum(rating_item_diff, axis=0)[:20])
        sim = rating_item_diff.T.dot(rating_item_diff) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

print('计算归一化的相似度矩阵...')
user_similarity_norm = cal_similarity_norm(ratings, kind='user')
item_similarity_norm = cal_similarity_norm(ratings, kind='item')
print('计算完成.')


# In[1071]:

def predict_norm_CF(user, item, k=20):
    '''baseline + item-item + '''
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    choice = nzero[item_similarity_norm[item, nzero].argsort()[::-1][:k]]
    prediction = (ratings[user, choice] - baseline[choice]).dot(item_similarity_norm[item, choice])                / sum(item_similarity_norm[item, choice]) + baseline[item]
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction 

print('载入测试集...')
test_df = pd.read_csv(testset_file, sep='\t', names=names)
test_df.head()
predictions = []
targets = []
print('测试集大小为 %d' % len(test_df))
print('采用归一化矩阵...')
k = 20
print('选取的K值为%d.' % k)
for row in test_df.itertuples():
    user, item, actual = row[1]-1, row[2]-1, row[3]
    predictions.append(predict_norm_CF(user, item, k))
    targets.append(actual)

print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[ ]:



