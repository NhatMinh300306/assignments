from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (p[j] - 1) * X[i]
            else:
                dW[:, j] += p[j] * X[i]

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

    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # 1. Tính ma trận điểm số (scores) có shape: (N, C)
    scores = X.dot(W)
    
    # Trừ đi giá trị lớn nhất trên từng hàng để tránh lỗi tràn số (numerical stability)
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # 2. Tính xác suất Softmax cho tất cả các mẫu (shape: N, C)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 3. Tính loss của các nhãn đúng (Cross-entropy loss)
    # Sử dụng advanced indexing để lấy ra xác suất của class đúng cho từng mẫu
    correct_log_probs = -np.log(probs[np.arange(num_train), y])
    
    # Tính trung bình cộng loss kèm theo thành phần L2 Regularization
    loss = np.sum(correct_log_probs) / num_train
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Sao chép ma trận xác suất để tính toán ma trận hệ số đạo hàm (dscores)
    dscores = probs.copy()
    
    # Với các class đúng (j == y_i), hệ số đạo hàm là (p_j - 1)
    dscores[np.arange(num_train), y] -= 1
    
    # 4. Tính dW bằng phép nhân ma trận giữa X chuyển vị và dscores
    # X.T có shape (D, N), dscores có shape (N, C) -> dW có shape (D, C)
    dW = X.T.dot(dscores)
    
    # Chia trung bình cho số lượng mẫu và cộng thêm đạo hàm của L2 Regularization (2 * reg * W)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
