import argparse
import os
import timeit
from importlib import import_module

import numpy as np
import scipy.misc
import tensorflow as tf
from attacks.minmax import minmax_uni
from utils import CIFAR10, MNIST, Experiment, Logger, str2bool

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="mnist", type=str, help="dataset: mnist/cifar")
parser.add_argument("--model", default="A", type=str, help="Model Architecture: A/B/C")
parser.add_argument("--eps", default=0.2, type=float, help="maximum distortion")
parser.add_argument("--norm", default=np.inf, type=float, help="order of norm")
parser.add_argument("--avg_case", default=False, type=str2bool, help="avg loss for multiple models")
parser.add_argument("--epochs", default=20, type=int, help="the maximum epoch to run minmax attack")
parser.add_argument("--alpha", default=6, type=float, help="1/alpha is step size for outer min")
parser.add_argument("--beta", default=50, type=float, help="1/beta is step size for inner max")
parser.add_argument("--gamma", default=4, type=float, help="regularization coefficient to balance avg/worst-case")
parser.add_argument("--loss_func", default="cw", type=str, help="loss function: xent/cw")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for training & testing")
parser.add_argument("--K", default=10, type=int, help="universal perturbation against K images")
parser.add_argument("--normalize", default=False, type=str2bool, help="whether to normalize the gradients")
parser.add_argument("--save_adv", default=False, type=str2bool, help="whether to save adv examples")
parser.add_argument("--num_pairs", default=2000, type=str2bool, help="number of pairs to save")
parser.add_argument("--group_by_labels", default=True, type=str2bool, help="group images by labels")
parser.add_argument("--sort_by_w_var", default=True, type=str2bool, help="sort images by the variance of learned w")


def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc, avg_acc = 0, 0, 0

    for batch in range(n_batch):
        # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc, batch_avg_acc = sess.run(
            [env.loss, env.acc, env.avg_acc], feed_dict={env.x: X_data[start:end], env.y: y_data[start:end]}
        )

        loss += batch_loss * cnt
        acc += batch_acc * cnt
        avg_acc += batch_avg_acc * cnt

    loss /= n_sample
    acc /= n_sample
    avg_acc /= n_sample

    # print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc, avg_acc


def make_uni_minmax(sess, env, X_data, K, epochs=1, eps=0.01, batch_size=128):
    """
    Generate MM by running env.x_uni.
    """
    print("\nMaking adversarials via minmax optimization")

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)
    w, h, c = X_data.shape[2:]
    Delta = np.zeros(shape=(n_sample, w, h, c))
    Ws = np.ones(shape=(n_sample, K)) / K

    for batch in range(n_batch):
        print(" batch {0}/{1}".format(batch + 1, n_batch), end="\r")
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        xadvs, delta, W = sess.run(
            [env.x_mm_uni, env.delta, env.W],
            feed_dict={env.x: X_data[start:end], env.uni_mm_eps: eps, env.uni_mm_epochs: epochs},
        )
        X_adv[start:end] = xadvs
        Delta[start:end] = delta
        Ws[start:end] = W
    print()

    return X_adv, Delta, Ws


def save_adv_examples(args, W, x_adv, x_delta, K, sort_by_var=False):
    # Saving universarial perturbations and adversarial examples
    num_pairs = args.num_pairs
    if sort_by_var:
        # Sort x_adv and delta by W's variance
        index = W.var(axis=1).argsort()[::-1]
        x_adv_ = x_adv[index][:num_pairs]
        x_delta_ = x_delta[index][:num_pairs]
        W_ = W[index][:num_pairs]

    x_adv_ = x_adv[0:num_pairs]
    x_delta_ = x_delta[:num_pairs]
    W_ = W[:num_pairs]
    print(x_adv_.shape, x_delta_.shape, W_.shape)

    if args.norm == np.inf:
        Lp = str("L_infty")
    else:
        Lp = "L_" + str(args.norm)
    save_dir = "save_uni/" + args.dataset + "/" + Lp

    for i in range(num_pairs):
        adv_save_dir = os.path.join(save_dir, str(i))
        if not os.path.exists(adv_save_dir):
            os.makedirs(adv_save_dir)

        if args.save_adv:
            for j in range(K):
                filename = str(j) + ".jpg"
                adv_filename = "adv_" + str(j) + ".jpg"
                img = x_adv_[i, j] - x_delta_[i]
                adv_img = x_adv_[i, j]
                if len(img.shape) == 2 or img.shape[-1] == 1:
                    img = np.stack([np.squeeze(img)] * 3, axis=-1)
                    adv_img = np.stack([np.squeeze(adv_img)] * 3, axis=-1)
                scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(os.path.join(adv_save_dir, filename))
                scipy.misc.toimage(adv_img, cmin=0.0, cmax=1.0).save(os.path.join(adv_save_dir, adv_filename))

            adv_delta = x_delta_[i]
            if len(adv_delta.shape) == 2 or adv_delta.shape[-1] == 1:
                adv_delta = np.stack([np.squeeze(adv_delta)] * 3, axis=-1)
            scipy.misc.toimage(adv_delta, cmin=0.0, cmax=1.0).save(os.path.join(adv_save_dir, "delta.jpg"))
            scipy.misc.toimage(-adv_delta, cmin=0.0, cmax=1.0).save(os.path.join(adv_save_dir, "delta_.jpg"))

        if args.save_w:
            with open(adv_save_dir + "/weights", "w") as f:
                f.write(str(W_[i]) + "\n")


def run(args, env, data, modelzoo, logger, K):
    # Preparing MNIST or CIFAR-10 data
    _, _, x_test, y_test = data.x_train, data.y_train, data.x_test, data.y_test

    # Grouping images and labels
    if args.dataset == "mnist":
        x_test = np.reshape(x_test, (-1, K, 28, 28, 1))
    else:
        x_test = np.reshape(x_test, (-1, K, 32, 32, 3))

    y_test = np.reshape(y_test, (-1, K, 10))
    print(x_test.shape, y_test.shape)

    # Initializing model architecture from modelzoo
    model_suffix = args.model
    model = getattr(modelzoo, "model" + model_suffix)

    # Constructing static computation graph (tf 1.x)
    if args.dataset == "mnist":
        env.x = tf.placeholder(tf.float32, (None, K, 28, 28, 1), name="x")
    else:
        env.x = tf.placeholder(tf.float32, (None, K, 32, 32, 3), name="x")
    env.y = tf.placeholder(tf.float32, (None, K, 10), name="y")
    env.training = tf.placeholder_with_default(False, (), name="mode")

    with tf.variable_scope("model" + model_suffix, reuse=tf.AUTO_REUSE):
        env.ybar, logits = [], []
        for i in range(K):
            ybar_single, logits_single = model(env.x[:, i, ...], logits=True, training=env.training)
            env.ybar.append(ybar_single)
            logits.append(logits_single)

        env.ybar = tf.stack(env.ybar, axis=1)
        logits = tf.stack(logits, axis=1)

    with tf.variable_scope("acc"):
        count = tf.equal(
            tf.argmax(tf.reshape(env.y, (-1, 10)), axis=1), tf.argmax(tf.reshape(env.ybar, (-1, 10)), axis=1)
        )
        count = tf.reduce_any(tf.reshape(count, (-1, K)), axis=1)
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name="acc")

        count_avg = tf.equal(
            tf.argmax(tf.reshape(env.y, (-1, 10)), axis=1), tf.argmax(tf.reshape(env.ybar, (-1, 10)), axis=1)
        )
        env.avg_acc = tf.reduce_mean(tf.cast(count_avg, tf.float32), name="avg_acc")

    with tf.variable_scope("loss"):
        env.loss = tf.losses.softmax_cross_entropy(
            tf.reshape(env.y, (-1, 10)), tf.reshape(logits, (-1, 10)), label_smoothing=0.1, weights=1.0
        )

    # Building min-max optimization graph
    env.uni_mm_eps = tf.placeholder(tf.float32, (), name="uni_mm_eps")
    env.uni_mm_epochs = tf.placeholder(tf.int32, (), name="uni_mm_epochs")

    env.x_mm_uni, env.delta, env.W = minmax_uni(
        model,
        env.x,
        norm=args.norm,
        epochs=env.uni_mm_epochs,
        eps=env.uni_mm_eps,
        fixed_w=args.avg_case,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        loss_func=args.loss_func,
        arch=args.model,
        normalize=args.normalize,
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
    x_adv, x_delta, W = make_uni_minmax(
        sess, env, x_test, args.K, eps=args.eps, epochs=args.epochs, batch_size=args.batch_size
    )
    stop = timeit.default_timer()
    logger.print("running time: " + str(stop - start))

    # Evaluating universal perturbations on multiple images
    adv_loss, adv_acc, adv_avg_acc = evaluate(sess, env, x_adv, y_test, batch_size=args.batch_size)

    logger.print("adv_loss     (eps@0.2): " + str(adv_loss))
    logger.print("adv_acc      (eps@0.2): " + str(adv_acc))
    logger.print("adv_avg_acc  (eps@0.2): " + str(adv_avg_acc))
    logger.print("success_one  (eps@0.2): " + str(1 - adv_avg_acc))
    logger.print("success_uni  (eps@0.2): " + str(1 - adv_acc))

    # save adv examples
    save_adv_examples(args, W, x_adv, x_delta, K, sort_by_var=args.sort_by_var)


def main():
    env = Experiment()
    args = parser.parse_args()

    modelzoo = import_module("models")
    assert args.dataset in ["mnist", "cifar"]

    if args.dataset == "mnist":
        data = MNIST(group_by_labels=args.group_by_labels)
    else:
        data = CIFAR10()

    logger = Logger(name="uni_model", base="./logs_uni/" + args.dataset)
    logger.print(args)
    run(args, env, data, modelzoo, logger, args.K)


if __name__ == "__main__":
    main()
