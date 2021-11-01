import numpy as np
import tensorflow as tf
from utils import proj_box, proj_box_appro, proj_prob_simplex

__all__ = [
    "minmax_ens",  # ensemble attack over multiple models: grad_delta
    "minmax_ens_logits",  # ensemble attack over multiple models: weigh logits
    "minmax_uni",  # universal perturbation over multiple images
    "minmax_eot",  # robust attack over multiple data transformation
]


def cw_loss(labels, logits, confidence=50):
    correct_logit = tf.reduce_sum(labels * logits, axis=1)
    wrong_logit = tf.reduce_max(labels * logits - 1e4 * labels, axis=1)
    loss = tf.nn.relu(correct_logit - wrong_logit + confidence)
    return loss


def minmax_ens(
    models,
    x,
    norm=np.inf,
    eps=0.2,
    epochs=20,
    alpha=7,
    beta=40,
    gamma=3,
    clip_min=0.0,
    clip_max=1.0,
    loss_func="xent",
    models_suffix="ABC",
    fixed_W=False,
    initial_w=None,
    appro=False,
    normalize=False,
):
    """
    Minmax Robust Optimization method.

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    if norm not in [np.inf, int(0), int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 0, 1, or 2.")

    eps = tf.abs(eps)
    K = len(models)
    # xadv = tf.identity(x)
    # delta = tf.zeros_like(x)
    delta = tf.random_normal(tf.shape(x), mean=0.0, stddev=0.1, seed=2019)
    batch_size = tf.shape(x)[0]
    if initial_w is None:
        W = tf.ones(shape=(batch_size, K)) / K
    else:
        # TODO: test tf.tile for ? dimension
        initial_w = tf.convert_to_tensor(initial_w)
        W = tf.tile(tf.expand_dims(initial_w, axis=1), [batch_size, 1])

    targets = []
    for i in range(K):
        with tf.variable_scope("model" + models_suffix[i], reuse=tf.AUTO_REUSE):
            ybar = models[i](x)
            indices = tf.argmax(ybar, axis=1)
            ydim = ybar.get_shape().as_list()[1]
            target = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)
            targets.append(target)

    if loss_func == "xent":
        # untargeted attack (minimize -cross-entropy)
        loss_fn = -tf.nn.softmax_cross_entropy_with_logits
    elif loss_func == "cw":
        loss_fn = cw_loss
    else:
        print("Unknown loss function. Defaulting to cross-entropy")
        loss_fn = -tf.nn.softmax_cross_entropy_with_logits

    def _update_F(delta, W):
        f = []
        for i, model in enumerate(models):
            # update loss values
            with tf.variable_scope("model" + models_suffix[i], reuse=tf.AUTO_REUSE):
                ybar, logits = model(x + delta, logits=True)
            # print("clipping loss")
            # f.append(tf.maximum(loss_fn(labels=targets[i], logits=logits), tf.constant(50.0)))
            # print(f)

        return tf.stack(f, axis=1)

    def _outer_min(delta, W, alpha=50):
        print("outer min...")
        F = _update_F(delta, W)
        loss_weighted = tf.reduce_sum(tf.multiply(W, F), axis=1)
        grad = tf.gradients(loss_weighted, delta)[0]

        # normalize the gradients
        if normalize:
            tol = 1e-8
            if norm == np.inf:
                grad = tf.sign(grad)
            elif norm == 1:
                ind = tuple(range(1, len(x.shape)))
                grad = grad / (tf.reduce_sum(np.abs(grad), axis=ind, keep_dims=True) + tol)
            elif norm == 2:
                ind = tuple(range(1, len(x.shape)))
                grad = grad / (tf.sqrt(tf.reduce_sum(tf.square(grad), axis=ind, keep_dims=True)) + tol)

        delta = tf.stop_gradient(delta - 1.0 / alpha * grad)
        if not appro:
            # analytical solution
            delta = proj_box(delta, norm, eps, -x, 1 - x)
        else:
            # approximate solution
            delta = proj_box_appro(delta, eps, norm)
            xadv = x + delta
            xadv = tf.clip_by_value(xadv, clip_min, clip_max)
            delta = xadv - x

        return delta

    def _inner_max(delta, W, gamma, beta):
        if fixed_W:
            return W
        print("inner max...")
        F = _update_F(delta, W)
        G = F - gamma * (W - 1 / K)
        W += 1.0 / beta * G
        W = proj_prob_simplex(W, batch_size, K)
        print(W)
        return W

    def _cond(delta, W, i):
        return tf.less(i, epochs)

    def _body(delta, W, i):
        delta = _outer_min(delta, W, alpha)
        W = _inner_max(delta, W, gamma, beta)
        return delta, W, i + 1

    delta, W, _ = tf.while_loop(_cond, _body, (delta, W, 0), back_prop=False, name="minmax_ens")

    return x + delta, W


def minmax_ens_logits(
    models,
    x,
    y=None,
    norm=np.inf,
    eps=0.2,
    epochs=20,
    alpha=50,
    beta=100,
    gamma=1,
    clip_min=0.0,
    clip_max=1.0,
    targeted=False,
    loss_func="xent",
    models_suffix="ABC",
    fixed_W=False,
):
    """
    Minmax Robust Optimization method.

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    if norm not in [np.inf, int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

    eps = tf.abs(eps)
    K = len(models)
    xadv = tf.identity(x)
    batch_size = tf.shape(x)[0]
    W = tf.ones(shape=(batch_size, K)) / K

    targets = []
    for i in range(K):
        with tf.variable_scope("model" + models_suffix[i], reuse=tf.AUTO_REUSE):
            ybar, logits = models[i](x, logits=True)
            ydim = ybar.get_shape().as_list()[1]
            indices = tf.argmax(ybar, axis=1)
            target = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)

            targets.append(target)

            print(tf.stack([W[:, i]] * logits.shape[1], axis=1))
            print(logits)
            if i == 0:
                logits_ = logits * tf.stack([W[:, i]] * logits.shape[1], axis=1)
            else:
                logits_ += logits * tf.stack([W[:, i]] * logits.shape[1], axis=1)

    ybar = tf.nn.softmax(logits_, name="ybar")
    ydim = ybar.get_shape().as_list()[1]

    if not targeted:
        # Using model predictions as ground truth to avoid label leaking
        indices = tf.argmax(ybar, axis=1)
    else:
        if y is None:
            print("targeted least-like")
            indices = tf.argmin(ybar, axis=1)
        else:
            indices = tf.cond(
                tf.equal(0, tf.rank(y)),
                lambda: tf.zeros([batch_size], dtype=tf.int32) + y,
                lambda: tf.zeros([batch_size], dtype=tf.int32),
            )

    target = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)

    if loss_func == "xent":
        loss_fn = -tf.nn.softmax_cross_entropy_with_logits
    elif loss_func == "cw":
        loss_fn = cw_loss
    else:
        print("Unknown loss function. Defaulting to cross-entropy")
        loss_fn = -tf.nn.softmax_cross_entropy_with_logits

    def _update_F(xadv, W):
        f = []
        for i, model in enumerate(models):
            # update loss values
            with tf.variable_scope("model" + models_suffix[i], reuse=tf.AUTO_REUSE):
                _, logits = model(xadv, logits=True)
                if i == 0:
                    logits_ = logits * tf.stack([W[:, i]] * logits.shape[1], axis=1)
                else:
                    logits_ += logits * tf.stack([W[:, i]] * logits.shape[1], axis=1)
            f.append(loss_fn(labels=targets[i], logits=logits))

        return logits_, tf.stack(f, axis=1)

    def _outer_min(xadv, W, alpha):
        print("outer min...")
        logits_, _ = _update_F(xadv, W)
        loss_weighted = loss_fn(labels=target, logits=logits_)

        if targeted:
            loss_weighted = -1.0 * loss_weighted

        grad = tf.gradients(loss_weighted, xadv)[0]

        tol = 1e-8
        if norm == np.inf:
            grad = tf.sign(grad)
        elif norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (tf.reduce_sum(np.abs(grad), axis=ind, keep_dims=True) + tol)
        elif norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (tf.sqrt(tf.reduce_sum(tf.square(grad), axis=ind, keep_dims=True)) + tol)

        xadv = tf.stop_gradient(xadv + 1.0 / alpha * grad)
        delta = proj_box(xadv - x, eps, norm)
        xadv = x + delta
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        print(delta)

        return xadv

    def _inner_max(xadv, W, gamma, beta):
        if fixed_W:
            return W
        print("inner max...")
        _, F = _update_F(xadv, W)
        G = F - gamma * (W - 1 / K)
        W += 1.0 / beta * G
        W = proj_prob_simplex(W, batch_size, K)
        print(W)
        return W

    def _cond(xadv, W, i):
        return tf.less(i, epochs)

    def _body(xadv, W, i):
        xadv = _outer_min(xadv, W, alpha)
        W = _inner_max(xadv, W, gamma, beta)
        return xadv, W, i + 1

    xadv, W, _ = tf.while_loop(_cond, _body, (xadv, W, 0), back_prop=False, name="minmax_ens_logits")

    return xadv, W


def minmax_uni(
    model,
    xs,
    norm=np.inf,
    eps=0.2,
    epochs=20,
    alpha=50,
    beta=100,
    gamma=10,
    clip_min=0.0,
    clip_max=1.0,
    loss_func="xent",
    arch="A",
    fixed_W=False,
    normalize=False,
):
    """
    Min-max universarial adversarial perturbations via APGDA.

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    if norm not in [np.inf, int(0), int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

    eps = tf.abs(eps)
    batch_size = tf.shape(xs)[0]
    K = xs.get_shape().as_list()[1]
    delta = tf.zeros_like(xs[:, 0])
    W = tf.ones(shape=(batch_size, K)) / K

    targets = []
    for i in range(K):
        with tf.variable_scope("model" + arch, reuse=tf.AUTO_REUSE):
            ybar = model(xs[:, i])
            indices = tf.argmax(ybar, axis=1)
            ydim = ybar.get_shape().as_list()[1]
            target = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)
            targets.append(target)

    if loss_func == "xent":
        loss_fn = tf.nn.softmax_cross_entropy_with_logits
    elif loss_func == "cw":
        loss_fn = cw_loss
    else:
        print("Unknown loss function. Defaulting to cross-entropy")
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    def _update_F(delta, W):
        f = []
        for i in range(K):
            # update loss values
            with tf.variable_scope("model" + arch, reuse=tf.AUTO_REUSE):
                _, logits = model(xs[:, i] + delta, logits=True)
            f.append(loss_fn(labels=targets[i], logits=logits))

        return tf.stack(f, axis=1)

    def _outer_min(delta, W, alpha=50):
        print("outer min...")
        F = _update_F(delta, W)
        print(W, F)
        loss_weighted = tf.reduce_sum(tf.multiply(W, F), axis=1)
        print(loss_weighted)

        grad = tf.gradients(loss_weighted, delta)[0]
        print(grad)

        if normalize:
            tol = 1e-8
            if norm == np.inf:
                grad = tf.sign(grad)
            elif norm == 1:
                ind = tuple(range(1, len(x.shape)))
                grad = grad / (tf.reduce_sum(np.abs(grad), axis=ind, keep_dims=True) + tol)
            elif norm == 2:
                ind = tuple(range(1, len(x.shape)))
                grad = grad / (tf.sqrt(tf.reduce_sum(tf.square(grad), axis=ind, keep_dims=True)) + tol)

        delta = tf.stop_gradient(delta + 1.0 / alpha * grad)

        for i in range(K):
            delta = proj_box(delta, norm, eps, -xs[:, i], 1 - xs[:, i])

        return delta

    def _inner_max(delta, W, gamma, beta):
        if fixed_W:
            return W
        print("inner max...")
        F = _update_F(delta, W)
        G = -F - gamma * (W - 1 / K)
        # G = F - gamma * (W - 1/K)
        W += 1.0 / beta * G
        print(W)
        W = proj_prob_simplex(W, batch_size, K)
        return W

    def _cond(delta, W, i):
        return tf.less(i, epochs)

    def _body(delta, W, i):
        delta = _outer_min(delta, W, alpha)
        W = _inner_max(delta, W, gamma, beta)
        return delta, W, i + 1

    delta, W, _ = tf.while_loop(_cond, _body, (delta, W, 0), back_prop=False, name="minmax_uni")

    return xs + tf.stack([delta] * K, axis=1), delta, W


def minmax_eot(
    model,
    x,
    G,
    norm=np.inf,
    eps=0.2,
    epochs=20,
    alpha=50,
    beta=1e5,
    gamma=10,
    Lambda=0.5,
    clip_min=0.0,
    clip_max=1.0,
    loss_func="xent",
    arch="A",
    fixed_W=False,
    normalize=False,
):
    """
    Min-max expectation over data transformation (EOT) via APGDA.

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    if norm not in [np.inf, int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

    eps = tf.abs(eps)
    K = len(G)
    # xadv = tf.identity(x)
    # delta = tf.zeros_like(x)
    delta = tf.random_normal(tf.shape(x), mean=0.0, stddev=0.1)
    batch_size = tf.shape(x)[0]
    W = tf.ones(shape=(batch_size, K)) / K

    targets = []
    for i in range(K):
        with tf.variable_scope("model" + arch, reuse=tf.AUTO_REUSE):
            ybar = model(G[i](x))
            indices = tf.argmax(ybar, axis=1)
            ydim = ybar.get_shape().as_list()[1]
            target = tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0)
            targets.append(target)

    if loss_func == "xent":
        loss_fn = -tf.nn.softmax_cross_entropy_with_logits
    elif loss_func == "cw":
        loss_fn = cw_loss
    else:
        print("Unknown loss function. Defaulting to cross-entropy")
        loss_fn = -tf.nn.softmax_cross_entropy_with_logits

    def _update_F(delta, W):
        f = []
        for i in range(K):
            # update loss values
            with tf.variable_scope("model" + arch, reuse=tf.AUTO_REUSE):
                _, logits = model(G[i](x + delta), logits=True)
            f.append(
                loss_fn(labels=targets[i], logits=logits) + 0.5 * Lambda * tf.norm(G[i](x + delta) - G[i](x), ord=2)
            )

        return tf.stack(f, axis=1)

    def _outer_min(delta, W, alpha=50):
        print("outer min...")
        F = _update_F(delta, W)
        loss_weighted = tf.reduce_sum(tf.multiply(W, F), axis=1)
        grad = tf.gradients(loss_weighted, delta)[0]

        if normalize:
            tol = 1e-8
            if norm == np.inf:
                grad = tf.sign(grad)
            elif norm == 1:
                ind = tuple(range(1, len(x.shape)))
                grad = grad / (tf.reduce_sum(np.abs(grad), axis=ind, keep_dims=True) + tol)
            elif norm == 2:
                ind = tuple(range(1, len(x.shape)))
                grad = grad / (tf.sqrt(tf.reduce_sum(tf.square(grad), axis=ind, keep_dims=True)) + tol)

        delta = tf.stop_gradient(delta + 1.0 / alpha * grad)
        delta = proj_box(delta, norm, eps, -x, 1 - x)
        """
        delta = proj_box(delta, eps, norm)
        xadv = x + delta
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        delta = xadv - x
        """
        return delta

    def _inner_max(delta, W, gamma, beta):
        if fixed_W:
            return W
        print("inner max...")
        F = _update_F(delta, W)
        G = F - gamma * (W - 1 / K)
        W += 1.0 / beta * G
        W = proj_prob_simplex(W, batch_size, K)
        return W

    def _cond(delta, W, i):
        return tf.less(i, epochs)

    def _body(delta, W, i):
        delta = _outer_min(delta, W, alpha)
        W = _inner_max(delta, W, gamma, beta)
        return delta, W, i + 1

    delta, W, _ = tf.while_loop(_cond, _body, (delta, W, 0), back_prop=False, name="minmax_eot")

    return x + delta, delta, W
