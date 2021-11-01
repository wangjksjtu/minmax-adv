import numpy as np
import tensorflow as tf


def proj_box_appro(v, eps, p):
    """
    Project the values in `v` on the L_p norm ball of size `eps`.
    :param v: Array of perturbations to clip.
    :type v: `np.ndarray`
    :param eps: Maximum norm allowed.
    :type eps: `float`
    :param p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
    :type p: `int`
    :return: Values of `v` after projection.
    :rtype: `np.ndarray`
    """

    # Pick a small scalar to avoid division by 0
    tol = 1e-8
    W, H, C = v.get_shape().as_list()[1:]
    v_ = tf.reshape(v, (tf.shape(v)[0], W * H * C))

    if p == 2:
        v_ = v_ * tf.expand_dims(tf.minimum(1.0, eps / (tf.linalg.norm(v_, axis=1) + tol)), axis=1)
    elif p == 1:
        v_ = v_ * tf.expand_dims(tf.minimum(1.0, eps / (tf.linalg.norm(v_, axis=1, ord=1) + tol)), axis=1)
    elif p == np.inf:
        v_ = tf.sign(v_) * tf.minimum(abs(v_), eps)
    else:
        raise NotImplementedError("Values of `p` different from 1, 2 and `np.inf` are currently not supported.")

    v = tf.reshape(v_, [tf.shape(v)[0], W, H, C])

    return v


def proj_box(v, p, eps, c, d):
    W, H, C = v.get_shape().as_list()[1:]
    N = tf.shape(v)[0]
    dim = W * H * C
    v = tf.reshape(v, (N, dim))
    c = tf.reshape(c, (N, dim))
    d = tf.reshape(d, (N, dim))

    clip_v = tf.minimum(v, d)
    clip_v = tf.maximum(clip_v, c)

    if p == np.inf:
        v = tf.clip_by_value(clip_v, -eps, eps)

    elif p == 0:
        e = tf.square(v)
        updates = tf.cast(tf.less(v, c), tf.float32)
        e = e - tf.multiply(updates, tf.square(v - c))
        updates = tf.cast(tf.less(d, v), tf.float32)
        e = e - tf.multiply(updates, tf.square(v - d))

        e_th = tf.contrib.nn.nth_element(e, tf.cast(eps, tf.int32), reverse=True)
        v = tf.multiply(clip_v, 1 - tf.cast(tf.less(e, tf.stack([e_th] * dim, axis=1)), tf.float32))

    elif p == 1:

        def bi_norm1(v, c, d, max_iter=5):
            lam_l = tf.zeros(shape=(N,))
            lam_r = tf.reduce_max(tf.abs(v), axis=1) - eps / dim

            def _cond_v(lam_l, lam_r, j):
                return tf.less(j, max_iter)

            def _body_v(lam_l, lam_r, j):
                lam = (lam_l + lam_r) / 2.0
                eq = (
                    tf.norm(
                        tf.maximum(
                            tf.minimum(tf.sign(v) * tf.maximum(tf.abs(v) - tf.stack([lam] * dim, axis=1), 0), d), c
                        ),
                        ord=1,
                        axis=1,
                    )
                    - eps
                )

                updates = tf.cast(tf.less(eq, 0.0), tf.float32)
                lam_r = lam_r - tf.multiply(updates, lam_r) + tf.multiply(updates, lam)

                updates = tf.cast(tf.less(0.0, eq), tf.float32)
                lam_l = lam_l - tf.multiply(updates, lam_l) + tf.multiply(updates, lam)

                return lam_l, lam_r, j + 1

            lam_l, lam_r, _ = tf.while_loop(_cond_v, _body_v, (lam_l, lam_r, 0), back_prop=False, name="bisection_lam")
            lam = (lam_l + lam_r) / 2.0

            return tf.maximum(tf.minimum(tf.sign(v) * tf.maximum(tf.abs(v) - tf.stack([lam] * dim, axis=1), 0), d), c)

        v = tf.cond(
            tf.reduce_all(tf.less(tf.norm(clip_v, ord=1, axis=1), eps)), lambda: clip_v, lambda: bi_norm1(v, c, d)
        )

    elif p == 2:

        def bi_norm2(v, c, d, max_iter=5):
            lam_l = tf.zeros(shape=(N,))
            lam_r = tf.norm(clip_v, ord=2, axis=1) / eps - 1

            def _cond_v(lam_l, lam_r, j):
                return tf.less(j, max_iter)

            def _body_v(lam_l, lam_r, j):
                lam = (lam_l + lam_r) / 2.0
                eq = (
                    tf.norm(
                        tf.maximum(tf.minimum(tf.divide(v, tf.stack([lam + 1] * dim, axis=1)), d), c), ord=2, axis=1
                    )
                    - eps
                )

                updates = tf.cast(tf.less(eq, 0.0), tf.float32)
                lam_r = lam_r - tf.multiply(updates, lam_r) + tf.multiply(updates, lam)

                updates = tf.cast(tf.less(0.0, eq), tf.float32)
                lam_l = lam_l - tf.multiply(updates, lam_l) + tf.multiply(updates, lam)

                return lam_l, lam_r, j + 1

            lam_l, lam_r, _ = tf.while_loop(_cond_v, _body_v, (lam_l, lam_r, 0), back_prop=False, name="bisection_lam")
            lam = (lam_l + lam_r) / 2.0

            return tf.maximum(tf.minimum(tf.divide(v, tf.stack([lam + 1] * dim, axis=1)), d), c)

        v = tf.cond(
            tf.reduce_all(tf.less(tf.norm(clip_v, ord=2, axis=1), eps)), lambda: clip_v, lambda: bi_norm2(v, c, d)
        )
    else:
        raise NotImplementedError("Values of `p` different from 0, 1, 2 and `np.inf` are currently not supported.")

    v = tf.reshape(v, [N, W, H, C])

    return v


def proj_prob_simplex(W, batch_size, K):
    W = bisection_mu(W, batch_size, K)
    return tf.maximum(tf.zeros(shape=(batch_size, K)), W)


def bisection_mu(W, batch_size, K, max_iter=20):
    mu_l = tf.reduce_min(W, axis=1) - 1 / K
    mu_r = tf.reduce_max(W, axis=1) - 1 / K
    mu = (mu_l + mu_r) / 2.0
    eq = tf.reduce_sum(tf.maximum(W - tf.stack([mu] * K, axis=1), 0), axis=1) - tf.ones(shape=(batch_size,))

    def _cond_mu(mu_l, mu_r, j):
        return tf.less(j, max_iter)

    def _body_mu(mu_l, mu_r, j):
        mu = (mu_l + mu_r) / 2.0
        eq = tf.reduce_sum(tf.maximum(W - tf.stack([mu] * K, axis=1), 0), axis=1) - tf.ones(shape=(batch_size,))
        updates = tf.cast(tf.less(eq, 0.0), tf.float32)
        mu_r = mu_r - tf.multiply(updates, mu_r) + tf.multiply(updates, mu)
        updates = tf.cast(tf.less(0.0, eq), tf.float32)
        mu_l = mu_l - tf.multiply(updates, mu_l) + tf.multiply(updates, mu)
        return mu_l, mu_r, j + 1

    mu_l, mu_r, _ = tf.while_loop(_cond_mu, _body_mu, (mu_l, mu_r, 0), back_prop=False, name="bisection_mu")

    mu = (mu_l + mu_r) / 2.0
    W -= tf.stack([mu] * K, axis=1)

    return W


def normalize_grad(grad, norm, tol=1e-8):
    if norm == np.inf:
        grad = tf.sign(grad)
    elif norm == 1:
        ind = tuple(range(1, len(x.shape)))
        grad = grad / (tf.reduce_sum(np.abs(grad), axis=ind, keep_dims=True) + tol)
    elif norm == 2:
        ind = tuple(range(1, len(x.shape)))
        grad = grad / (tf.sqrt(tf.reduce_sum(tf.square(grad), axis=ind, keep_dims=True)) + tol)
    
    return grad