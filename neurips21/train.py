import argparse
import os
import time
from importlib import import_module

import numpy as np
import tensorflow as tf
from attacks.fast_gradient import fgm
from utils import CIFAR10, MNIST, Logger

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist", type=str, help="Dataset: mnist/cifar")
parser.add_argument("--epochs", default=250, type=int, help="Epochs for training models")
parser.add_argument("--model", default="A", type=str, help="Model architecture: A/B/C")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size training models")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer")
parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate")
parser.add_argument("--lr_decay", default=0.1, type=float, help="Learning rate decay")
parser.add_argument("--scheduler", default="100,175", type=str, help="Learning rate scheduler")
args = parser.parse_args()

DATASET = args.dataset
MODEL = "model" + args.model
EPOCHS = args.epochs
batch_size = args.batch_size
weight_decay = args.weight_decay
optim = args.optimizer.lower()
initial_lr = args.lr
scheduler = [int(x) for x in args.scheduler.split(",")]
lr_decay = args.lr_decay

models = import_module("models")
assert DATASET in ["mnist", "cifar"]
assert optim in ["sgd", "adam"]

if DATASET == "mnist":
    data = MNIST()
    assert MODEL[-1] in ["A", "B", "C", "D"]
else:
    data = CIFAR10(augument=False)

fmt = {"train_loss": ".4f", "train_acc": ".4f", "val_loss": ".4f", "val_acc": ".4f"}
logger = Logger(fmt=fmt, name=MODEL, base="./logs/" + DATASET)


def evaluate(sess, env, X_data, y_data, batch_size=256, dest="test"):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc], feed_dict={env.x: X_data[start:end], env.y: y_data[start:end], env.training: False}
        )
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    merged = sess.run(env.merged, feed_dict={env.loss_ph: loss, env.acc_ph: acc})
    if dest == "train":
        env.train_writer.add_summary(merged, env.epoch)
    else:
        env.test_writer.add_summary(merged, env.epoch)
    # print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))

    return loss, acc


def train(
    sess,
    env,
    X_data,
    y_data,
    X_val=None,
    y_val=None,
    epochs=1,
    load=False,
    shuffle=True,
    batch_size=128,
    name="modelA",
    dataset="mnist",
):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, "saver"):
            return print("\nError: cannot find saver op")
        print("\nLoading saved model")
        return env.saver.restore(sess, "models/" + dataset + "/{}".format(name))

    print("\nTrain model")
    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)

    if hasattr(env, "saver"):
        os.makedirs("models", exist_ok=True)

    max_acc = 0
    for epoch in range(epochs):
        # print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))
        env.epoch = epoch
        if shuffle:
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            # print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end], env.y: y_data[start:end], env.training: True})

        if X_val is not None:
            train_loss, train_acc = evaluate(sess, env, X_data, y_data, dest="train")
            val_loss, val_acc = evaluate(sess, env, X_val, y_val, dest="test")
            logger.add_scalar(epoch, "train_loss", train_loss)
            logger.add_scalar(epoch, "train_acc", train_acc)
            logger.add_scalar(epoch, "val_loss", val_loss)
            logger.add_scalar(epoch, "val_acc", val_acc)
            logger.iter_info()

        if max_acc < val_acc:
            print("\nSaving model {:4f} -> {:4f}".format(max_acc, val_acc))
            max_acc = val_acc
            save_dir = "models/" + dataset
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            env.saver.save(sess, save_dir + "/{}".format(name))

    logger.print("")
    logger.print("val_acc: %4f" % max_acc)


def predict(sess, env, X_data, batch_size=128):
    """
    Conduct inference by running env.ybar.
    """
    print("\nPredicting")
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(" batch {0}/{1}".format(batch + 1, n_batch), end="\r")
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print("\nMaking adversarials via FGSM")

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(" batch {0}/{1}".format(batch + 1, n_batch), end="\r")
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgsm, feed_dict={env.x: X_data[start:end], env.fgsm_eps: eps, env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


def main():
    x_train, y_train, x_test, y_test = data.x_train, data.y_train, data.x_test, data.y_test
    model = getattr(models, MODEL)

    if DATASET == "mnist":
        env.x = tf.placeholder(tf.float32, (None, 28, 28, 1), name="x")
    else:
        env.x = tf.placeholder(tf.float32, (None, 32, 32, 3), name="x")
    env.y = tf.placeholder(tf.float32, (None, 10), name="y")
    env.training = tf.placeholder_with_default(False, (), name="mode")

    env.global_step = tf.Variable(0, trainable=False)
    n_batch = int(np.ceil(x_train.shape[0] / float(batch_size)))

    boundaries = [x * n_batch for x in scheduler]
    values = [initial_lr * lr_decay ** (n + 1) for n in range(len(scheduler))]
    values.insert(0, initial_lr)
    env.lr = tf.train.piecewise_constant(env.global_step, boundaries, values)

    with tf.variable_scope(MODEL):
        env.ybar, logits = model(env.x, logits=True, training=env.training)

        with tf.variable_scope("acc"):
            count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
            env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name="acc")

        with tf.variable_scope("loss"):
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y, logits=logits)
            env.loss = tf.reduce_mean(xent, name="loss")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope("train_op"):
            if optim == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=env.lr)
            elif optim == "sgd":
                optimizer = tf.train.MomentumOptimizer(learning_rate=env.lr, momentum=0.9, use_nesterov=True)
            else:
                raise NotImplementedError
            costs = []
            for var in tf.trainable_variables():
                costs.append(tf.nn.l2_loss(var))

        with tf.control_dependencies(update_ops):
            # env.train_op = optimizer.minimize(env.loss)
            env.train_op = optimizer.minimize(env.loss + weight_decay * tf.add_n(costs), global_step=env.global_step)

        env.saver = tf.train.Saver()

    with tf.variable_scope(MODEL, reuse=True):
        env.fgsm_eps = tf.placeholder(tf.float32, (), name="fgsm_eps")
        env.fgsm_epochs = tf.placeholder(tf.int32, (), name="fgsm_epochs")
        env.x_fgsm = fgm(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

    # Collecting summary
    with tf.name_scope("summary"):
        env.loss_ph = tf.placeholder(tf.float32, shape=None, name="loss")
        tf.summary.scalar("loss", env.loss_ph)
        env.acc_ph = tf.placeholder(tf.float32, shape=None, name="accuracy")
        tf.summary.scalar("accuracy", env.acc_ph)
        tf.summary.scalar("learning rate", env.lr)
        env.merged = tf.summary.merge_all()

    # Initializing graph
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Tensorboard summary writer
    summary_dir = "logs_tb/" + DATASET + "/" + MODEL
    env.train_writer = tf.summary.FileWriter(os.path.join(summary_dir, "train"), sess.graph)
    env.test_writer = tf.summary.FileWriter(os.path.join(summary_dir, "eval"), sess.graph)

    # Training
    t_begin = time.time()
    train(
        sess,
        env,
        x_train,
        y_train,
        x_test,
        y_test,
        load=False,
        epochs=EPOCHS,
        name=MODEL,
        dataset=DATASET,
        batch_size=batch_size,
    )
    seconds = time.time() - t_begin
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    logger.print("total_time: %d:%02d:%02d" % (h, m, s))

    # Generating adversarial data via FGSM
    x_adv = make_fgsm(sess, env, x_test, eps=0.2, epochs=1)
    adv_loss, adv_acc = evaluate(sess, env, x_adv, y_test)
    logger.print("")
    logger.print("adv_loss (fgsm@0.2): " + str(adv_loss))
    logger.print("adv_acc (fgsm@0.2): " + str(adv_acc))


if __name__ == "__main__":
    main()
