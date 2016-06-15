import var
import numpy as np

def cal_sparsity():
    '''计算矩阵密度'''
    sparsity = float(len(var.ratings.nonzero()[0]))
    sparsity /= var.n_users * var.n_items
    sparsity *= 100
    return sparsity

def cal_mean():
    '''计算总体均值，各user打分均值，各item打分均值...'''
    var.all_mean = np.mean(var.ratings[var.ratings!=0])
    var.user_mean = sum(var.ratings.T) / sum((var.ratings!=0).T)
    var.item_mean = sum(var.ratings) / sum((var.ratings!=0))
    var.user_mean = np.where(np.isnan(var.user_mean), var.all_mean, var.user_mean)
    var.item_mean = np.where(np.isnan(var.item_mean), var.all_mean, var.item_mean)
    
def cal_similarity(kind, epsilon=1e-9):
    '''利用Cosine距离计算相似度'''
    '''epsilon: 防止Divide-by-zero错误，进行矫正'''
    if kind == 'user':
        sim = var.ratings.dot(var.ratings.T) + epsilon
    elif kind == 'item':
        sim = var.ratings.T.dot(var.ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def cal_similarity_norm(kind, epsilon=1e-9):
    '''采用归一化的指标:Pearson correlation coefficient'''
    if kind == 'user':
        # 对同一个user的打分归一化
        rating_user_diff = var.ratings.copy()
        for i in range(var.ratings.shape[0]):
            nzero = var.ratings[i].nonzero()
            rating_user_diff[i][nzero] = var.ratings[i][nzero] - var.user_mean[i]
        sim = rating_user_diff.dot(rating_user_diff.T) + epsilon
    elif kind == 'item':
        # 对同一个item的打分归一化
        rating_item_diff = var.ratings.copy()
        for j in range(var.ratings.shape[1]):
            nzero = var.ratings[:,j].nonzero()
            rating_item_diff[:,j][nzero] = var.ratings[:,j][nzero] - var.item_mean[j]
        sim = rating_item_diff.T.dot(rating_item_diff) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def rmse(pred, actual):
    '''计算预测结果的rmse'''
    from sklearn.metrics import mean_squared_error
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual))