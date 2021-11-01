import numpy as np
import tensorflow as tf
from .utils import proj_box_appro

__all__ = [
    "pgd",  # projected gradient descent
]


def pgd(model, x, norm=np.inf, eps=0.3, eps_step=0.1, epochs=10, clip_min=0.0, clip_max=1.0):
    """
    Projected gradient descent (PGD) method.

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    if norm not in [np.inf, int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

    xmin = x - eps
    xmax = x + eps

    xadv = tf.identity(x)

    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0),
    )

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    eps_step = tf.abs(eps_step)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        tol = 1e-8

        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        (grad,) = tf.gradients(loss, xadv)
        if norm == np.inf:
            grad = tf.sign(grad)
        elif norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (tf.reduce_sum(np.abs(grad), axis=ind, keep_dims=True) + tol)
        elif norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (tf.sqrt(tf.reduce_sum(tf.square(grad), axis=ind, keep_dims=True)) + tol)
        xadv = tf.stop_gradient(xadv + eps_step * grad)

        xadv = tf.maximum(tf.minimum(xadv, xmax), xmin)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)

        noise = proj_box_appro(xadv - x, eps, norm)
        xadv = x + noise

        return xadv, i + 1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False, name="pgd")
    return xadv
