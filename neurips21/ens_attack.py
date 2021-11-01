import argparse
import os
import timeit
from importlib import import_module

import numpy as np
import scipy.misc
import tensorflow as tf
from attacks.minmax import minmax_ens, minmax_ens_logits
from utils import CIFAR10, MNIST, Experiment, Logger, str2bool

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset", default="mnist", type=str, help="dataset: mnist/cifar")
parser.add_argument("--eps", default=0.2, type=float, help="maximum distortion")
parser.add_argument("--avg_case", default=False, type=str2bool, help="avg loss for multiple models")
parser.add_argument("--epochs", default=20, type=int, help="#iter for minmax attack")
parser.add_argument("--alpha", default=7, type=float, help="1/alpha is step size for outer min")
parser.add_argument("--beta", default=40, type=float, help="1/beta is step size for inner max")
parser.add_argument("--gamma", default=3, type=float, help="regularization coefficient to balance avg/worst-case")
parser.add_argument("--loss_func", default="cw", type=str, help="loss function: xent/cw")
parser.add_argument("--weigh_logits", default=False, type=str2bool, help="weigh logits rather than loss")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for training & testing")
parser.add_argument("--norm", default=np.inf, type=float, help="order of norm")
parser.add_argument("--models", default="ABC", type=str, help="models to ensemble")
parser.add_argument("--appro", default=False, type=str2bool, help="approximate solution to box constraint")
parser.add_argument("--normalize", default=False, type=str2bool, help="whether to normalize the gradients")
parser.add_argument("--save_w", default=False, type=str2bool, help="whether to save the weights wi")
parser.add_argument("--save_adv", default=False, type=str2bool, help="whether to save adv examples")


def get_models(modelzoo, archs):
    models = []
    for arch in archs:
        model = getattr(modelzoo, "model" + arch)
        models.append(model)
    return models


def evaluate(sess, env, X_data, y_data, K, batch_size=128):
    """Evaluate TF model by running env.loss and env.acc."""

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = np.zeros((K,)), np.zeros((K + 1,))

    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc], feed_dict={env.x: X_data[start:end], env.y: y_data[start:end]}
        )
        loss += batch_loss * cnt
        acc += batch_acc * cnt

    loss /= n_sample
    acc /= n_sample

    return loss, acc


def make_ens_minmax(sess, env, X_data, K, epochs=1, eps=0.01, batch_size=128):
    """
    Generate min-max adversarial attacks by running env.x_ens.
    """
    print("\nCreating ensemble attacks via min-max optimization")

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    W_final = np.zeros(shape=(n_sample, K))

    for batch in range(n_batch):
        print(" batch {0}/{1}".format(batch + 1, n_batch), end="\r")
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        xadv, W = sess.run(
            [env.x_mm_ens, env.W], feed_dict={env.x: X_data[start:end], env.ens_mm_eps: eps, env.ens_mm_epochs: epochs}
        )
        X_adv[start:end] = xadv
        W_final[start:end] = W

    return X_adv, W_final


def run(args, env, data, modelzoo, logger, K):
    # Preparing MNIST or CIFAR-10 data
    _, _, x_test, y_test = data.x_train, data.y_train, data.x_test, data.y_test

    # Constructing static computation graph (tf 1.x)
    if args.dataset == "mnist":
        env.x = tf.placeholder(tf.float32, (None, 28, 28, 1), name="x")
    else:
        env.x = tf.placeholder(tf.float32, (None, 32, 32, 3), name="x")
    env.y = tf.placeholder(tf.float32, (None, 10), name="y")
    env.training = tf.placeholder_with_default(False, (), name="mode")
    env.ybar, env.logits = [], []

    models = get_models(modelzoo, args.models)
    for i in range(K):
        with tf.variable_scope("model" + args.models[i]):
            ybar, logits = models[i](env.x, logits=True, training=env.training)
            env.ybar.append(ybar)
            env.logits.append(logits)

    env.acc = []
    with tf.variable_scope("acc"):
        count_or = None
        for i in range(K):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar[i], axis=1))
            env.acc.append(tf.reduce_mean(tf.cast(count, tf.float32), name="acc" + str(i)))
            if count_or is None:
                count_or = count
            else:
                count_or = tf.logical_or(count_or, count)
        env.acc.append(tf.reduce_mean(tf.cast(count_or, tf.float32), name="acc"))

    env.loss = []
    with tf.variable_scope("loss"):
        for i in range(K):
            env.loss.append(tf.losses.softmax_cross_entropy(env.y, env.logits[i], label_smoothing=0.1, weights=1.0))

    env.ybar = tf.stack(env.ybar, axis=0)
    env.logits = tf.stack(env.logits, axis=0)
    env.acc = tf.stack(env.acc, axis=0)
    env.loss = tf.stack(env.loss, axis=0)
    env.ens_mm_eps = tf.placeholder(tf.float32, (), name="ens_mm_eps")
    env.ens_mm_epochs = tf.placeholder(tf.int32, (), name="ens_mm_epochs")

    # Building min-max optimization graph
    if not args.weigh_logits:
        env.x_mm_ens, env.W = minmax_ens(
            models,
            env.x,
            norm=args.norm,
            epochs=env.ens_mm_epochs,
            eps=env.ens_mm_eps,
            fixed_W=args.avg_case,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            loss_func=args.loss_func,
            models_suffix=args.models,
            appro=args.appro,
            normalize=args.normalize,
        )
    else:
        env.x_mm_ens, env.W = minmax_ens_logits(
            models,
            env.x,
            norm=args.norm,
            epochs=env.ens_mm_epochs,
            eps=env.ens_mm_eps,
            fixed_W=args.avg_case,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            targeted=False,
            loss_func=args.loss_func,
            models_suffix=args.models,
        )

    # Initializing graph
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Loading model weights
    all_vars = tf.global_variables()
    env.saver = []
    for i in range(K):
        model_vars = [k for k in all_vars if k.name.startswith("model" + args.models[i])]
        env.saver.append(tf.train.Saver(model_vars))
        env.saver[-1].restore(sess, "models/" + args.dataset + "/model" + args.models[i])

    # Validating on benign data
    print("Validating on benign data")
    loss_benign, acc_benign = evaluate(sess, env, x_test, y_test, K, batch_size=args.batch_size)

    for i in range(K):
        logger.print("loss" + str(i) + ": " + str(loss_benign[i]) + "\t" + "acc" + str(i) + ": " + str(acc_benign[i]))

    # Generating adversarial examples via minmax optimization
    start = timeit.default_timer()
    x_adv, w_final = make_ens_minmax(sess, env, x_test, K, eps=args.eps, epochs=args.epochs, batch_size=args.batch_size)
    stop = timeit.default_timer()
    logger.print("running time: " + str(stop - start))

    # Evaluating adversarial examples on multiple models
    adv_loss, adv_acc = evaluate(sess, env, x_adv, y_test, K, batch_size=args.batch_size)
    for i in range(K):
        logger.print("adv_loss" + str(i) + ": " + str(adv_loss[i]) + "\t" + "adv_acc" + str(i) + ": " + str(adv_acc[i]))
    logger.print("adv_acc_or: " + str(adv_acc[K]))
    logger.print("foolrate: " + str(1 - adv_acc[K]))

    # Saving domain weights
    if args.norm == np.inf:
        Lp = str("L_infty")
    else:
        Lp = "L_" + str(args.norm)

    if args.save_w:
        w_save_dir = "save/" + args.dataset + "/weights/" + Lp
        if not os.path.exists(w_save_dir):
            os.makedirs(w_save_dir)
        w_save_path = os.path.join(w_save_dir, "w_" + str(epochs))
        np.save(w_save_path, w_final)
        print("weights saved to " + w_save_path)

    # Saving robust adversarial examples
    if args.save_adv:
        adv_save_dir = "save/" + args.dataset + "/advs/" + Lp
        for i in range(10):
            if not os.path.exists(os.path.join(adv_save_dir, str(i))):
                os.makedirs(os.path.join(adv_save_dir, str(i)))

        for i in range(x_test.shape[0]):
            adv_img = x_adv[i]
            y_label = np.argmax(y_test[i])
            filename = str(y_label) + "/" + str(i) + ".jpg"
            if len(adv_img.shape) == 2 or adv_img.shape[-1] == 1:
                adv_img = np.stack([np.squeeze(adv_img)] * 3, axis=-1)
            scipy.misc.toimage(adv_img, cmin=0.0, cmax=1.0).save(os.path.join(adv_save_dir, filename))
        print("adv examples saved to " + adv_save_dir)


def main():
    env = Experiment()
    args = parser.parse_args()
    K = len(args.models)

    if args.norm != np.inf:
        args.norm = int(args.norm)

    modelzoo = import_module("models")
    assert args.dataset in ["mnist", "cifar"]

    if args.dataset == "mnist":
        data = MNIST()
    else:
        data = CIFAR10()

    logger = Logger(name="ens_model", base="./logs/" + args.dataset)
    logger.print(args)
    run(args, env, data, modelzoo, logger, K)


if __name__ == "__main__":
    main()
