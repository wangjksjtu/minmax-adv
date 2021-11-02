import numpy as np
import tensorflow as tf
from .utils import proj_box, proj_box_appro, proj_prob_simplex, normalize_grad

__all__ = [
    "minmax_ens",  # ensemble attack over multiple models
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
    fixed_w=False,
    initial_w=None,
    appro_proj=False,
    normalize=False,
    rand_init=True,
):
    """
    Min-max ensemble attack over via APGDA.

    :param models: A list of victim models that return the logits.
    :param x: The input placeholder.
    :param norm: The L_p norm: np.inf, 2, 1, 0.  
    :param eps: The scale factor for noise.
    :param epochs: Number of epochs to run attack generation.
    :param alpha: 1/alpha is step size for outer minimization (update perturation).
    :param beta: 1/beta is step size for inner maximization (update domain weights).
    :param gamma: regularization coefficient to balance avg-case and worst-case.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :param loss_func: Adversarial loss function: "xent" or "cw".
    :param models_suffix: The suffixes for victim models, e.g., "ABC" for three models.
    :param fixed_w: Fixing domain weights and does not update.
    :param initial_w: The initial domain weight vector.
    :param appro_proj: Use approximate solutions for box constraint (l_p) projection.
    :param normalize: Normalize the gradients when optimizing perturbations.
    :param rand_init: Random normal initialization for perturbations.

    :return: A tuple of tensors, contains adversarial samples for each input, 
             and converged domain weights.
    """
    if norm not in [np.inf, int(0), int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 0, 1, or 2.")

    eps = tf.abs(eps)
    K = len(models)
    if rand_init:
        delta = tf.random_normal(tf.shape(x), mean=0.0, stddev=0.1)
    else:
        delta = tf.zeros_like(x)
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
            f.append(loss_fn(labels=targets[i], logits=logits))
            # print(f)

        return tf.stack(f, axis=1)

    def _outer_min(delta, W, alpha=50):
        print("outer min...")
        F = _update_F(delta, W)
        loss_weighted = tf.reduce_sum(tf.multiply(W, F), axis=1)
        grad = tf.gradients(loss_weighted, delta)[0]

        # normalize the gradients
        if normalize:
            grad = normalize_grad(grad, norm)

        delta = tf.stop_gradient(delta - 1.0 / alpha * grad)
        
        # project perturbations
        if not appro_proj:
            # analytical solution
            delta = proj_box(delta, norm, eps, clip_min - x, clip_max - x)
        else:
            # approximate solution
            delta = proj_box_appro(delta, eps, norm)
            xadv = x + delta
            xadv = tf.clip_by_value(xadv, clip_min, clip_max)
            delta = xadv - x

        return delta

    def _inner_max(delta, W, gamma, beta):
        if fixed_w:
            # average case or static heuristic weights
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

    delta, W, _ = tf.while_loop(_cond, _body, (delta, W, 0), back_prop=False, name="minmax_ens")

    return x + delta, W


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
    fixed_w=False,
    appro_proj=False,
    normalize=False,
):
    """
    Min-max universarial adversarial perturbations via APGDA.

    :param model: The victim model that returns the logits.
    :param xs: The input placeholder (?, K, H, W, C).
    :param norm: The L_p norm: np.inf, 2, 1, 0.
    :param eps: The scale factor for noise.
    :param epochs: Number of epochs to run attack generation.
    :param alpha: 1/alpha is step size for outer minimization (update perturation).
    :param beta: 1/beta is step size for inner maximization (update domain weights).
    :param gamma: regularization coefficient to balance avg-case and worst-case.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :param loss_func: Adversarial loss function: "xent" or "cw".
    :param models_suffix: The suffixes for victim models, e.g., "ABC" for three models.
    :param fixed_w: Fixing domain weights and does not update.
    :param appro_proj: Use approximate solutions for box constraint (l_p) projection.
    :param normalize: Normalize the gradients when optimizing perturbations.

    :return: A tuple of tensors, contains adversarial samples for each input, 
             universal perturbations and converged domain weights.
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
                _, logits = model(xs[:, i] + delta, logits=True)
            f.append(loss_fn(labels=targets[i], logits=logits))

        return tf.stack(f, axis=1)

    def _outer_min(delta, W, alpha=50):
        print("outer min...")
        F = _update_F(delta, W)
        loss_weighted = tf.reduce_sum(tf.multiply(W, F), axis=1)
        grad = tf.gradients(loss_weighted, delta)[0]
        if normalize:
            grad = normalize_grad(grad, norm)
        delta = tf.stop_gradient(delta + 1.0 / alpha * grad)
        for i in range(K):
            # project perturbations
            if not appro_proj:
                # analytical solution
                delta = proj_box(delta, norm, eps, clip_min - xs[:, i], clip_max - xs[:, i])
            else:
                # approximate solution
                delta = proj_box_appro(delta, eps, norm)
                xadv = xs[:, i] + delta
                xadv = tf.clip_by_value(xadv, clip_min, clip_max)
                delta = xadv - xs[:, i]
        return delta

    def _inner_max(delta, W, gamma, beta):
        if fixed_w:
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
    fixed_w=False,
    appro_proj=False,
    normalize=False,
    rand_init=True,
):
    """
    Min-max expectation over data transformation (EOT) via APGDA.

    :param model: The victim model that returns the logits.
    :param x: The input placeholder (?, H, W, C).
    :param G: A list of data transformer that return transformed inputs.
    :param norm: The L_p norm: np.inf, 2, 1, 0.
    :param eps: The scale factor for noise.
    :param epochs: Number of epochs to run attack generation.
    :param alpha: 1/alpha is step size for outer minimization (update perturation).
    :param beta: 1/beta is step size for inner maximization (update domain weights).
    :param gamma: regularization coefficient to balance avg-case and worst-case.
    :param Lambda: L2 regularization cofficient between transformed images and original one.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :param loss_func: Adversarial loss function: "xent" or "cw".
    :param arch: The model suffix, e.g., "A" for MLP.
    :param fixed_w: Fixing domain weights and does not update.
    :param appro_proj: Use approximate solutions for box constraint (l_p) projection.
    :param normalize: Normalize the gradients when optimizing perturbations.
    :param rand_init: Random normal initialization for perturbations.

    :return: A tuple of tensors, contains adversarial samples for each input, 
             universal perturbations and converged domain weights.
    """
    if norm not in [np.inf, int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

    eps = tf.abs(eps)
    K = len(G)
    if rand_init:
        delta = tf.random_normal(tf.shape(x), mean=0.0, stddev=0.1)
    else:
        delta = tf.zeros_like(x)
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
            grad = normalize_grad(grad, norm)
        delta = tf.stop_gradient(delta + 1.0 / alpha * grad)
        if not appro_proj:
            delta = proj_box(delta, norm, eps, clip_min - x, clip_max - x)
        else:
            delta = proj_box_appro(delta, eps, norm)
            xadv = x + delta
            xadv = tf.clip_by_value(xadv, clip_min, clip_max)
            delta = xadv - x

        return delta

    def _inner_max(delta, W, gamma, beta):
        if fixed_w:
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
