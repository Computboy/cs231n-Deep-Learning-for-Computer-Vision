from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    """
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        # shape: (C,) 标签数量
        scores -= np.max(scores)
        # 数值稳定：减去分数最高位置的元素，防止指数爆炸[现在最大数字为e^0]
        
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        # probability：输出Softmax的一系列概率

        loss += -np.log(probs[y[i]])
        # 损失函数取-log值

        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (probs[j] - 1) * X[i]
            else:
                dW[:, j] += probs[j] * X[i]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = X @ W # 矩阵乘法，形状(N,C)
    scores -= np.max(scores, axis=1, keepdims=1)
    # 按行减去最大值（防止数值溢出）
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=1)
    # 按行计算指数和
    probs = exp_scores / sum_exp_scores
    # 得出概率
    correct_probs = probs[np.arange(num_train), y]
    # 按高级索引，拿到每行（就是每张图片）对应正确标签下的概率，并将其写入一个数组中
    loss = np.sum(-np.log(correct_probs)) / num_train
    # 损失函数是每行Softmax输出的负对数之和，再除以样本数
    probs[np.arange(num_train), y] -= 1

    dW = X.T @ probs
    dW /= num_train

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW
