from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        kNN分类器的训练：只是将训练数据储存到了self.X_train和self.y_train中，仅此而已

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
             输入的X就是测试样本
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]              # 获取第一维度的长度，样本的行数（X:测试样本数量）
        num_train = self.X_train.shape[0]  # 获取第一维度的长度，不过是已经储存在类中的数据（训练样本）
        dists = np.zeros((num_test, num_train))   # 矩阵中元素赋予全零值
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # 简单来说，就是计算第i个测试样本和第j个训练样本之间的l2距离，并储存至矩阵dists的[i][j]号元素中
                distance = pow(self.X_train[j] - X[i], 2)  # 得到的还是一个向量
                dists[i][j] = np.sum(distance)  # 需要使用各维度上数字相加得到L2距离(的平方)
                pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            diff = self.X_train - X[i, :]  # numpy的广播机制 (N_train, D) - (1, D) = (N_train, D)
            sq_diff = diff ** 2
            sum_sq_diff = np.sum(sq_diff, axis=1)  # 按行求和，得到一个一维数组，可以被添加至矩阵的某行/列中
            dists[i, :] = sum_sq_diff
            pass
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # 解题思路详见NoteBook
        test_square = np.sum(X ** 2, axis=1, keepdims=1)   # 每行求和，并保留二位特性，方便Broadcast
        train_square = np.sum(self.X_train ** 2, axis=1, keepdims=1).T # 由于训练数据是按列填充的，因此需要转置
        matrix_product = X @ self.X_train.T   #矩阵乘法，(i,j)元素即为dist(i,j)分量
        diff = test_square + train_square - 2 * matrix_product
        # diff = np.sqrt(np.maximum(sq_dists, 0.0))
        dists = diff
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            kst_near = np.argsort(dists[i])[:k]  # 储存训练数据的索引值
            closest_y = self.y_train[kst_near]   # 储存训练数据对应的标签

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            labels, counts = np.unique(closest_y, return_counts=True)   # 记录索引
            max_count = np.max(counts)
            # 筛选出所有出现次数等于max_count的标签
            candidate_labels = labels[counts == max_count]
            # 选最小的标签（处理平局）
            y_pred[i] = np.min(candidate_labels)

        return y_pred
