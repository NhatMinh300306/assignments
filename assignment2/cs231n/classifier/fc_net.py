from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deterministic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        layer_dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            W_key = f"W{i + 1}"
            b_key = f"b{i + 1}"
            self.params[W_key] = np.random.normal(
                loc=0.0, scale=weight_scale, size=(layer_dims[i], layer_dims[i + 1])
            )
            self.params[b_key] = np.zeros(layer_dims[i + 1])

            if self.normalization in ["batchnorm", "layernorm"] and i < self.num_layers - 1:
                gamma_key = f"gamma{i + 1}"
                beta_key = f"beta{i + 1}"
                self.params[gamma_key] = np.ones(layer_dims[i + 1])
                self.params[beta_key] = np.zeros(layer_dims[i + 1])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = {}
        out = X

        # Forward pass cho các lớp ẩn: {affine - [batch/layer norm] - relu - [dropout]} x (L - 1)
        for i in range(1, self.num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            caches[i] = {}

            # 1. Affine layer
            out, caches[i]["affine"] = affine_forward(out, W, b)

            # 2. Normalization layer
            if self.normalization == "batchnorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                out, caches[i]["norm"] = batchnorm_forward(
                    out, gamma, beta, self.bn_params[i - 1]
                )
            elif self.normalization == "layernorm":
                gamma = self.params[f"gamma{i}"]
                beta = self.params[f"beta{i}"]
                out, caches[i]["norm"] = layernorm_forward(
                    out, gamma, beta, self.bn_params[i - 1]
                )

            # 3. ReLU nonlinearity
            out, caches[i]["relu"] = relu_forward(out)

            # 4. Dropout layer
            if self.use_dropout:
                out, caches[i]["dropout"] = dropout_forward(
                    out, self.dropout_param
                )

        # Forward pass cho lớp output cuối cùng (chỉ dùng affine)
        W_last = self.params[f"W{self.num_layers}"]
        b_last = self.params[f"b{self.num_layers}"]
        caches[self.num_layers] = {}
        scores, caches[self.num_layers]["affine"] = affine_forward(out, W_last, b_last)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if mode == "test":
            return scores

        loss, grads = 0.0, {}

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 1. Tính Softmax loss và gradient ban đầu
        loss, dout = softmax_loss(scores, y)

        # Cộng L2 regularization loss cho toàn bộ weights
        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(W * W)

        # 2. Backward pass cho lớp output cuối cùng
        last_cache = caches[self.num_layers]["affine"]
        dout, dW, db = affine_backward(dout, last_cache)
        grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db

        # 3. Backward pass ngược dần qua các lớp ẩn
        for i in range(self.num_layers - 1, 0, -1):
            # Dropout backward
            if self.use_dropout:
                dout = dropout_backward(dout, caches[i]["dropout"])

            # ReLU backward
            dout = relu_backward(dout, caches[i]["relu"])

            # Normalization backward
            if self.normalization == "batchnorm":
                dout, dgamma, dbeta = batchnorm_backward(
                    dout, caches[i]["norm"]
                )
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
            elif self.normalization == "layernorm":
                dout, dgamma, dbeta = layernorm_backward(
                    dout, caches[i]["norm"]
                )
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta

            # Affine backward
            dout, dW, db = affine_backward(dout, caches[i]["affine"])
            grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
            grads[f"b{i}"] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads