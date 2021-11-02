import argparse
import os
import timeit
from importlib import import_module

import numpy as np
import tensorflow as tf
from attacks.minmax import minmax_eot
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar", type=str, help="dataset: mnist/cifar")
parser.add_argument("--model", default="A", type=str, help="Model Architecture: A/B/C")
parser.add_argument("--eps", default=0.025, type=float, help="maximum distortion")
parser.add_argument("--avg_case", default=False, type=str2bool, help="avg loss for multiple models")
parser.add_argument("--epochs", default=20, type=int, help="the maximum epoch to run minmax attack")
parser.add_argument("--alpha", default=6, type=float, help="1/alpha is step size for outer min")
parser.add_argument("--beta", default=50, type=float, help="1/beta is step size for inner max")
parser.add_argument("--gamma", default=3, type=float, help="regularization coefficient to balance avg/worst-case")
parser.add_argument("--Lambda", default=0.5, type=float, help="data transformation regularizer")
parser.add_argument("--loss_func", default="cw", type=str, help="loss function: xent/cw")
parser.add_argument("--batch_size", default=1024, type=int, help="batch size for training & testing")
parser.add_argument("--K", default=6, type=int, help="perturbation against K transformations")


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc, avg_acc, all_acc = 0, 0, 0, np.zeros(K)

    for batch in range(n_batch):
        # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc, batch_avg_acc, batch_all_acc = sess.run(
            [env.loss, env.acc, env.avg_acc, env.all_acc],
            feed_dict={env.x: X_data[start:end], env.y: np.stack([y_data[start:end]] * K, axis=1)},
        )
        loss += batch_loss * cnt
        acc += batch_acc * cnt
        avg_acc += batch_avg_acc * cnt
        all_acc += batch_all_acc * cnt

    loss /= n_sample
    acc /= n_sample
    avg_acc /= n_sample
    all_acc /= n_sample

    return loss, acc, avg_acc, all_acc


def make_minmax_eot(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate MM by running env.x_trans.
    """
    print("\nMaking adversarials via minmax optimization")

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(" batch {0}/{1}".format(batch + 1, n_batch), end="\r")
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        xadvs, delta, W = sess.run(
            [env.x_mm_trans, env.delta, env.W],
            feed_dict={env.x: X_data[start:end], env.trans_mm_eps: eps, env.trans_mm_epochs: epochs},
        )
        print(W[0:10])
        X_adv[start:end] = xadvs
    print()

    return X_adv


def run(args, env, data, modelzoo, G, logger, K):
    # Preparing MNIST or CIFAR-10 data
    _, _, x_test, y_test = data.x_train, data.y_train, data.x_test, data.y_test

    if args.dataset == "mnist":
        env.x = tf.placeholder(tf.float32, (None, 28, 28, 1), name="x")
    else:
        env.x = tf.placeholder(tf.float32, (None, 32, 32, 3), name="x")
    env.y = tf.placeholder(tf.float32, (None, K, 10), name="y")
    env.training = tf.placeholder_with_default(False, (), name="mode")

    # Initializing model architecture from modelzoo
    model_suffix = args.model
    model = getattr(modelzoo, "model" + model_suffix)

    # Constructing static computation graph (tf 1.x)
    with tf.variable_scope("model" + model_suffix, reuse=tf.AUTO_REUSE):
        env.ybar, logits = [], []
        for i in range(K):
            ybar_single, logits_single = model(G[i](env.x), logits=True, training=env.training)
            env.ybar.append(ybar_single)
            logits.append(logits_single)

        env.ybar = tf.stack(env.ybar, axis=1)
        logits = tf.stack(logits, axis=1)

    with tf.variable_scope("acc"):
        count = tf.equal(
            tf.argmax(tf.reshape(env.y, (-1, 10)), axis=1), tf.argmax(tf.reshape(env.ybar, (-1, 10)), axis=1)
        )
        count_any = tf.reduce_any(tf.reshape(count, (-1, K)), axis=1)
        env.acc = tf.reduce_mean(tf.cast(count_any, tf.float32), name="acc")

        count_avg = tf.equal(
            tf.argmax(tf.reshape(env.y, (-1, 10)), axis=1), tf.argmax(tf.reshape(env.ybar, (-1, 10)), axis=1)
        )
        env.avg_acc = tf.reduce_mean(tf.cast(count_avg, tf.float32), name="avg_acc")

        count_all = tf.cast(tf.reshape(count, (-1, K)), tf.float32)
        env.all_acc = tf.reduce_mean(count_all, axis=0, name="all_acc")

    with tf.variable_scope("loss"):
        env.loss = tf.losses.softmax_cross_entropy(
            tf.reshape(env.y, (-1, 10)), tf.reshape(logits, (-1, 10)), label_smoothing=0.1, weights=1.0
        )

    # Building min-max optimization graph
    env.trans_mm_eps = tf.placeholder(tf.float32, (), name="trans_mm_eps")
    env.trans_mm_epochs = tf.placeholder(tf.int32, (), name="trans_mm_epochs")

    env.x_mm_trans, env.delta, env.W = minmax_eot(
        model,
        env.x,
        G,
        norm=np.inf,
        epochs=env.trans_mm_epochs,
        eps=env.trans_mm_eps,
        fixed_w=args.avg_case,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        Lambda=args.Lambda,
        loss_func=args.loss_func,
        arch=model_suffix,
    )

    # Initializing graph
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Loading model weights
    all_vars = tf.global_variables()
    model_vars = [k for k in all_vars if k.name.startswith("model" + model_suffix)]
    env.saver = tf.train.Saver(model_vars)
    env.saver.restore(sess, "models/" + args.dataset + "/model" + model_suffix)

    # Generating adversarial data via minmax optimization
    start = timeit.default_timer()
    x_adv = make_minmax_eot(sess, env, x_test, eps=args.eps, epochs=args.epochs, batch_size=args.batch_size)
    stop = timeit.default_timer()

    logger.print("running time: " + str(stop - start))

    adv_loss, adv_acc, adv_avg_acc, adv_all_acc = evaluate(sess, env, x_adv, y_test, batch_size=args.batch_size)

    logger.print("adv_loss      (eps@%.3f): %.2f" % (eps, adv_loss))
    logger.print("adv_acc       (eps@%.3f): %.2f" % (eps, adv_acc))
    logger.print("adv_avg_acc   (eps@%.3f): %.2f" % (eps, adv_avg_acc))
    logger.print("adv_all_acc   (eps@%.3f): %.2f" % (eps) + str(adv_all_acc))
    logger.print("success_one   (eps@%.3f): %.4f" % (eps, 1 - adv_avg_acc))
    logger.print("success_trans (eps@%.3f): %.4f" % (eps, 1 - adv_acc))


def main():
    env = Experiment()
    args = parser.parse_args()

    assert args.dataset in ["mnist", "cifar"]
    modelzoo = import_module("models")

    trans_list1 = [
        identity,
        fliplr,
        flipud,
        adjust_brightness,
        # adjust_contrast (no gradient defined)
        adjust_gamma,
        crop_and_resize,
        rot60,
        # adjust_jpeg_quality (no suport for batch processing)
    ]
    trans_list2 = [identity, rot30, rot30_, rot60, rot60_, rot90, rot90_, rot120, rot120_, rot150, rot150_, rot180]
    G = trans_list1[:args.K]

    if args.dataset == "mnist":
        data = MNIST()
    else:
        data = CIFAR10()

    logger = Logger(name="trans_model", base="./logs/" + args.dataset)
    logger.print(args)
    logger.print(G)

    run(args, env, data, modelzoo, G, logger, args.K)


if __name__ == "__main__":
    main()
