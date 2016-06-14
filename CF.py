
# coding: utf-8

# In[669]:

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


# In[670]:

print('初始化变量...')
names = ['user_id', 'item_id', 'rating', 'timestamp']
trainingset_file = 'dataset/ml-100k/u3.base'
testset_file= 'dataset/ml-100k/u3.test'
n_users = 943
n_items = 1682
ratings = np.zeros((n_users, n_items))


# In[671]:

df = pd.read_csv(trainingset_file, sep='\t', names=names)
print('载入训练集...')
print('数据集样例为:')
print(df.head())
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
print('载入完成.')
print('打分矩阵规模为 %d*%d.' % (n_users, n_items))
print('测试集有效打分个数为 %d.' % len(df))


# In[672]:

# 计算矩阵密度
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print('测试集矩阵密度为: {:4.2f}%'.format(sparsity))
print()


# In[673]:

def rmse(pred, actual):
    '''计算预测结果的rmse'''
    from sklearn.metrics import mean_squared_error
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual))


# In[674]:

print('------ Naive算法(baseline) ------')


# In[675]:

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
    print('均值计算完成，总体打分均值为 %.2f' % all_mean)


# In[676]:

print('计算训练集各项统计数据...')
cal_mean()


# In[677]:

def predict_naive(user, item):
    prediction = item_mean[item] + user_mean[user] - all_mean
    return prediction


# In[678]:

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

print('测试结果的rmse为 %.2f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[679]:

print('------ item-item协同过滤算法(相似度未归一化) ------')


# In[680]:

def cal_similarity(ratings, kind, epsilon=1e-9):
    '''利用Cosine距离计算相似度'''
    '''epsilon: 防止Divide-by-zero错误，进行矫正'''
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


# In[681]:

print('计算相似度矩阵...')
user_similarity = cal_similarity(ratings, kind='user')
item_similarity = cal_similarity(ratings, kind='item')
print('计算完成.')


# In[682]:

def predict_itemCF(user, item, k=100):
    '''item-item协同过滤算法,预测rating'''
    nzero = ratings[user].nonzero()[0]
    prediction = ratings[user, nzero].dot(item_similarity[item, nzero])                / sum(item_similarity[item, nzero])
    return prediction


# In[683]:

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

print('测试结果的rmse为 %.2f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[684]:

print('------ 加了baseline的协同过滤算法(相似度未归一化) ------')


# In[685]:

def predict_itemCF_baseline(user, item, k=100):
    '''CF item-item算法,预测rating'''
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    prediction = (ratings[user, nzero] - baseline[nzero]).dot(item_similarity[item, nzero])                / sum(item_similarity[item, nzero]) + baseline[item]
    return prediction 


# In[686]:

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

print('测试结果的rmse为 %.2f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[687]:

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

print('测试结果的rmse为 %.2f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[688]:

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
    
print('测试结果的rmse为 %.2f' % rmse(np.array(predictions), np.array(targets)))
print()


# In[689]:

print('------ Todo: 归一化的协同过滤算法------')
print('------ Todo: Top k的协同过滤算法------' )

