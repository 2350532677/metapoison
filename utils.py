import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import subprocess

def count_available_gpus():
    return str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')


def _get_basename(name):
    name = '/'.join(name.split('/')[2:])
    return name.split(':')[0]


def _reshape_labels_like_logits(labels, logits, batchsize, nclass=10):
    return tf.reshape(tf.one_hot(labels, nclass), [batchsize, nclass])


def metrics(labels, logits, batchsize):
    with tf.variable_scope('metrics'):
        labels_reshaped = _reshape_labels_like_logits(labels, logits, batchsize)
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels_reshaped, logits=logits), name='xent')
        equal = tf.equal(labels, tf.cast(tf.argmax(logits, axis=1), dtype=labels.dtype))
        acc = tf.reduce_mean(tf.to_float(equal), name='acc')
    return xent, acc


def carlini(labels, logits, batchsize, clamp=-100):
    with tf.variable_scope('carlini'):
        labels_reshaped = _reshape_labels_like_logits(labels, logits, batchsize)
        labels_reshaped = tf.cast(labels_reshaped, dtype=logits.dtype)
        target_logit = tf.reduce_sum(logits * labels_reshaped)
        second_logit = tf.reduce_max(logits - logits * labels_reshaped)
        tmp = logits - logits * labels_reshaped
        return tf.maximum(second_logit - target_logit, clamp)  # , target_logit, second_logit, tmp


def count_params_in_scope():
    scope = tf.get_default_graph().get_name_scope()
    nparam = sum([np.prod(w.shape.as_list()) for w in tf.trainable_variables(scope)])
    # print('scope:', scope, '#params', nparam)
    return nparam


def imagesc(img, title=None, experiment=None, step=None, scale='minmax'):
    if scale == 'minmax':
        img = img - img.ravel().min()
        img = img / img.ravel().max()
    elif type(scale) is float or type(scale) is int:  # good for perturbations
        img = img * .5 / scale + .5
    elif type(scale) is list or type(scale) is tuple:  # good for images
        assert len(scale) == 2, 'scale arg must be length 2'
        lo, hi = scale
        img = (img - lo) / (hi - lo)
    plt.clf()
    plt.imshow(img)
    if title:
        plt.title(title)
    if experiment:
        experiment.log_figure(figure_name=title, step=step)


def pgdstep(img, grad, orig, stepsize=.01, epsilon=.08, perturb=False):
    if perturb: img += (np.random.rand(*img.shape) - .5) * 2 * epsilon
    img += stepsize * np.sign(grad)
    img = np.clip(img, orig - epsilon, orig + epsilon)
    img = np.clip(img, 0, 255)
    return img


def l2_weights(weights):
    return tf.add_n([tf.reduce_sum(weight ** 2) for weight in weights.values() if len(weight.shape.as_list()) > 1])


def tf_preprocess(inputs, batchsize):
    # preprocessing data augmentation
    inputs = tf.pad(inputs, [[0, 0], [4, 4], [4, 4], [0, 0]])
    inputs = tf.random_crop(inputs, [batchsize, 32, 32, 3])
    inputs = tf.map_fn(tf.image.random_flip_left_right, inputs)
    return inputs


def avg_n_dicts(dicts, experiment=None, step=None):
    # given a list of dicts with the same exact schema, return a single dict with same schema whose values are the
    # key-wise average over all input dicts
    means = {}
    for dic in dicts:
        for key in dic:
            if key not in means: means[key] = 0
            means[key] += dic[key] / len(dicts)
    if experiment is not None:
        experiment.log_metrics(means, step=step)
    return means


def merge_n_dicts(dicts):
    # given a list of dicts with mutually exclusive schema, return a dict of all key-value pairs merged
    out = {}
    for d in dicts:
        if d is not None:
            out.update(d)
    return out


def plot_dict_series(dict_series, prefix=None, experiment=None, step=None):
    # given a list of dicts with the same schema, make a series plot for each key in the schema
    # if dict_series is a list of list of dicts, then overlap all plots in the second nested list
    serialized = {}
    for i, timestep in enumerate(dict_series):
        if type(timestep) is dict: timestep = [timestep]
        for dic in timestep:
            for key, val in dic.items():
                if key not in serialized: serialized[key] = []
                if len(serialized[key]) <= i: serialized[key].append([])
                serialized[key][-1].append(val)
    for key, series in serialized.items():
        plt.clf()
        plt.plot(np.array(series))
        plt.title('step {}'.format(step))
        plt.ylabel(key)
        if experiment is not None: experiment.log_figure(figure_name='{}_{}'.format(prefix, key), step=step)


def copy_to_args_from_experiment(args, exptkey, api, attrs):
    # given a comet experiment and an args namespace, copy the values of all attributes in attrs from experiment to args
    for param in api.get_experiment_parameters(exptkey):
        # attrs is a list of attributes that you want to copy over
        if param['name'] in attrs:
            if type(getattr(args, param['name'])) is int: setattr(args, param['name'], int(param['valueCurrent']))
            if type(getattr(args, param['name'])) is str: setattr(args, param['name'], str(param['valueCurrent']))
            if type(getattr(args, param['name'])) is float: setattr(args, param['name'], float(param['valueCurrent']))
    return args


def has_exitflag(exptkey, api):
    # see whether experiment has logged exitflag via log_other
    return len(api.get_experiment_other(exptkey, 'exitflag')) > 0


def transpose_list_of_lists(l):
    return list(map(list, zip(*l)))


def set_available_gpus(args):
    if args.gpu is not None: os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
    else: args.gpu = list(range(count_available_gpus()))
    return args.gpu


class Dummy:
    def __getattribute__(self, attr):
        return lambda *arg, **kwarg: None

def lr_schedule(lrnrate, epoch, warmupperiod=5):
    warmupfactor = min(1, (epoch + 1) / warmupperiod)
    if epoch < 40:
        return 1e00 * lrnrate * warmupfactor
    elif epoch < 60:
        return 1e-1 * lrnrate * warmupfactor
    elif epoch < 80:
        return 1e-2 * lrnrate * warmupfactor
    else:
        return 1e-3 * lrnrate * warmupfactor
    
def appendfeats(feats, feat, victimfeed, ybase, ytarget):
    cleaninputs, cleanlabels = [value for key, value in victimfeed.items() if 'adapter-0/cleaninputs' in str(key)][0]
    cleanmask = [value for key, value in victimfeed.items() if 'cleanmask' in str(key)][0]
    poisonmask = [value for key, value in victimfeed.items() if 'poisonmask' in str(key)][0]
    npoison = sum(poisonmask)
    feats['targetfeats'] = feat[0]
    feats['targetlabels'] = ytarget
    feats['cleanfeats'].extend(feat[1][npoison:])
    feats['poisonfeats'].extend(feat[1][:npoison])
    feats['cleanlabels'].extend(cleanlabels)
    feats['poisonlabels'].extend(ybase[poisonmask])

def get_featdist(feats):
    targetfeats, poisonfeats = feats['targetfeats'], feats['poisonfeats']
    targetfeat = np.array(targetfeats[:1])
    poisonfeats = np.array(poisonfeats)
    featdist = np.mean(np.linalg.norm(poisonfeats - targetfeat, axis=1))
    return featdist
