import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow import tensorflow as tf
from utils import metrics, l2_weights, tf_preprocess, carlini
import pickle
import numpy as np
import tensorflow as tf

class Meta():

    def __init__(self, args, xbase, ybase, xtarget, ytarget, metaid=None):
        self.args = args
        self.xbase, self.ybase = xbase, ybase
        self.xtarget, self.ytarget = xtarget, ytarget
        # weights to randomly initialize, True means all weights
        self.coldstart_names = True if args.pretrain is None else ['w6', 'b6']
        self.trainable_names = True  # weights unfrozen during training, True means all weights
        self.metaid = metaid
        self.cached_weights, self.cached_poisons = {}, {}

        # change to 64 bit for more reproducible results
        self.floattype = tf.float64 if self.args.bit64 else tf.float32
        self.inttype = tf.int64 if self.args.bit64 else tf.int32

        # build graph
        self.build_metalearner_graph()
        self.coldstartop = tf.variables_initializer([self.weights0[name] for name in self.coldstart_names])

        # variables initializers. variables in adapter-1, adapter-2, etc are created by keras and unwanted
        unwanteda =  set(tf.global_variables('adapter')) - set(tf.global_variables('adapter-0'))
        unwantedt =  set(tf.global_variables('targeter')) - set(tf.global_variables('targeter-0'))
        unwanted = unwanteda.union(unwantedt)
        unwanted = set(u for u in unwanted if 'moving' not in u.name)
        global_variables = set(tf.global_variables()) - unwanted
        self.modified_global_initializer = tf.variables_initializer(list(global_variables))
        self.allcoldstartop = tf.variables_initializer(list(global_variables - set(tf.global_variables('poisons'))))

    def build_metalearner_graph(self):

        # define feeds
        self.lrnrate = tf.placeholder(name='lrnrate', shape=[], dtype=self.floattype)
        self.craftrate = tf.placeholder(name='craftrate', shape=[], dtype=self.floattype)
        self.augment = tf.constant(self.args.augment, dtype=tf.bool, name='augment')

        # define poisons and the boolean mask for minibatching
        with tf.variable_scope('poisons'):
            self.poisoninputs = tf.Variable(tf.constant(self.xbase, dtype=self.floattype), name='poisoninputs')
            self.poisonlabels = tf.Variable(tf.constant(self.ybase, dtype=self.inttype), name='poisonlabels')
            self.poisonmask = tf.constant([False] * self.args.npoison, dtype=tf.bool, name='poisonmask')
            poisoninputs_ = tf.boolean_mask(self.poisoninputs, self.poisonmask, axis=0, name='poisoninputs_masked')
            poisonlabels_ = tf.boolean_mask(self.poisonlabels, self.poisonmask, axis=0, name='poisonlabels_masked')

        # define target
        inputsT, labelsT = tf.constant(self.xtarget, dtype=self.floattype), tf.constant(self.ytarget, dtype=self.inttype)
        self.trains, self.cleanmasks, self.xents, self.accs, self.xentTs, self.accTs, self.cwTs = [], [], [], [], [], [], []

        # select network arcthiecture
        if self.args.tf111: from learners_compat import ConvNet, ResNet #, KerasModel # compatible with tf1.11
        else: from learners import ConvNet, ResNet #, KerasModel # these learners require tf1.14
        if self.args.net == 'ConvNet':
            net = ConvNet(self.args)
        elif self.args.net in ['ResNet']:
            net = ResNet(self.args, num_blocks=3, classes=10)
        elif self.args.net in ['ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2',
                               'ResNet101V2', 'ResNet152V2', 'ResNeXt50', 'ResNeXt101',
                               'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                               'InceptionResNetV2', 'InceptionV3', 'VGG16', 'VGG19', 'Xception', 'MobileNetV2', 'MobileNet']:
            net = KerasModel(self.args, architecture=self.args.net , data='CIFAR10')
        else:
            raise ValueError('Unknown network architecture.')

        # build metagraph
        for i in range(self.args.nadapt):
            with tf.variable_scope('adapter-' + str(i)):

                # inputs for feed_dict
                inputs = tf.placeholder(name='cleaninputs', shape=[
                                        self.args.batchsize, 32, 32, 3], dtype=self.floattype)
                labels = tf.placeholder(name='cleanlabels', shape=[self.args.batchsize], dtype=self.inttype)
                self.trains.append((inputs, labels))

                # inject poisons at first iteration
                if i == 0:
                    self.cleanmask = tf.constant([True] * self.args.batchsize, dtype=tf.bool, name='cleanmask')
                    cleaninputs_ = tf.boolean_mask(inputs, self.cleanmask, axis=0, name='cleaninputs_masked')
                    cleanlabels_ = tf.boolean_mask(labels, self.cleanmask, axis=0, name='cleanlabels_masked')
                    inputs = tf.concat([poisoninputs_, cleaninputs_], axis=0, name='concat_inputs')
                    labels = tf.concat([poisonlabels_, cleanlabels_], axis=0, name='concat_labels')
                if self.args.augment: # data augmentation
                    inputs = tf.cond(self.augment, lambda: tf_preprocess(inputs, self.args.batchsize), lambda: inputs)

                # forward pass
                if i == 0:  # construct weight variables (future unrolled weights are nonvariable tensors)
                    self.weights0 = net.construct_weights()
                    self.weights = self.weights0.copy()  # copy the list, but does not deeply copy the tensors
                    if self.trainable_names is True: self.trainable_names = list(
                        self.weights.keys())  # True means all weights
                    if self.coldstart_names is True: self.coldstart_names = list(self.weights.keys())
                logits, feats = net.forward(inputs, self.weights)
                xent, acc = metrics(labels, logits, self.args.batchsize)
                self.xents.append(xent)
                self.accs.append(acc)
                if i == 0: self.feat0 = feats
                if self.args.weightdecay: xent = xent + 2e-4 * l2_weights(self.weights)

                # compute fast weights
                grad_list = tf.gradients(xent, [self.weights[key] for key in self.trainable_names])
                gradients = dict(zip(self.trainable_names, grad_list))
                for key in gradients:
                    assert gradients[key] is not None, "Key {} has no gradient signal.".format(key)
                    self.weights[key] = self.weights[key] - self.lrnrate * gradients[key]

            with tf.variable_scope('targeter-' + str(i + 1)):
                # forward pass on target at current stage of unrolling
                logits, _ = net.forward(inputsT, self.weights)
                xentT, accT = metrics(labelsT, logits, self.args.ntarget)
                cwT = carlini(labelsT, logits, self.args.ntarget)
                self.xentTs.append(xentT)
                self.accTs.append(accT)
                self.cwTs.append(cwT)
                self.logits = logits

        self.trains = tuple(self.trains)

        with tf.variable_scope('targeter-0'):
            # target loss on current network without unrolling
            logits, self.featT0 = net.forward(inputsT, self.weights0)
            self.xentT0, self.accT0 = metrics(labelsT, logits, self.args.ntarget)
            self.cwT0 = carlini(labelsT, logits, self.args.ntarget)

        # average target loss over all adapt steps
        with tf.variable_scope('metaloss'):
            self.xentT = tf.add_n(self.xentTs, name='meta_xent') / len(self.xentTs)
            self.accT = tf.add_n(self.accTs, name='meta_acc') / len(self.accTs)
            self.cwT = tf.add_n(self.cwTs, name='meta_cw') / len(self.cwTs)
            self.objective = self.cwT if self.args.objective == 'cw' else self.xentT if self.args.objective == 'xent' else None

        with tf.variable_scope('metagrad'):
            # compute metagradient
            self.optim = tf.train.AdamOptimizer(self.craftrate)
            metagradients = self.optim.compute_gradients(self.objective, var_list=self.poisoninputs)
            metagrad, metavar = metagradients[0]
            self.metagrad_accum = tf.get_variable(
                name='metagrad_accum', shape=self.xbase.shape, dtype=self.floattype, initializer=tf.zeros_initializer, trainable=False)
            self.objective_accum = tf.get_variable(
                name='objective_accum', shape=[], dtype=self.floattype, initializer=tf.zeros_initializer, trainable=False)
            self.accumop = [tf.assign_add(self.metagrad_accum, metagrad, name='metagrad_accumop'),
                            tf.assign_add(self.objective_accum, self.objective, name='objective_accumop')]
            if self.args.horovod:
                import horovod.tensorflow as hvd
                self.avg_metagrads = hvd.allreduce(self.metagrad_accum)
            else:
                self.avg_metagrads = tf.placeholder(shape=list(self.metagrad_accum.get_shape()), dtype=self.floattype, name='avg_metagrads')

        with tf.variable_scope('craftop'):
            # apply metagradient
            self.avg_metagradients = [(self.avg_metagrads, self.poisoninputs), ]
            craftop = self.optim.apply_gradients(self.avg_metagradients, name='craftop')
            # pgd clipping
            with tf.control_dependencies([craftop]):
                clipped = tf.clip_by_value(self.poisoninputs, self.xbase - self.args.eps, self.xbase + self.args.eps)
                clipped = tf.clip_by_value(clipped, 0, 255)
                clipop = tf.assign(self.poisoninputs, clipped, name='clipop')
                # reset metagrad accumulator
                zeroop = tf.assign(self.metagrad_accum, tf.zeros_like(self.metagrad_accum), name='zeroop')
            self.craftop = [craftop, clipop, zeroop]

        with tf.variable_scope('trainop'):
            with tf.control_dependencies([self.xentT0, self.accT0, self.cwT0]):
                var_list = [value for key, value in self.weights0.items() if key in self.trainable_names]
                if self.args.optimizer == 'sgd': optim = tf.train.GradientDescentOptimizer(self.lrnrate)
                elif self.args.optimizer == 'mom': optim = tf.train.MomentumOptimizer(self.lrnrate, momentum=.9)
                else: optim = tf.train.MomentumOptimizer(self.lrnrate, momentum=.9)
                # else: raise ValueError('Invalid optimizer')
                self.trainop = [optim.minimize(self.xents[0], var_list=var_list)]

        with tf.variable_scope('debug'):
            # result dictionaries
            self.resultM = dict(xentT=self.xentT,
                                accT=self.accT,
                                cwT=self.cwT,
                                )
            self.resultL = dict(xentT0=self.xentT0,
                                accT0=self.accT0,
                                cwT0=self.cwT0,
                                xent=self.xents[0],
                                acc=self.accs[0],
                                )
            self.resultV = dict(xentV=self.xents[0],
                                accV=self.accs[0],
                                )
            self.feat = (self.featT0, self.feat0)

    def load_weights(self, sess, pretrain_weights):
        [self.weights0[key].load(pretrain_weights[key], sess) for key in pretrain_weights]

    def init_weights(self, sess, pretrain_weights=None):
        if pretrain_weights is not None: self.load_weights(sess, pretrain_weights)
        sess.run(self.coldstartop)

    def cache_weights(self, sess, cache='default', restore=False):
        if not restore:
            self.cached_weights[cache] = sess.run(self.weights0)
        else:
            [val.load(self.cached_weights[cache][key], sess) for key, val in self.weights0.items()]

    def cache_poison(self, sess, cache='default', restore=False):
        if not restore:
            self.cached_poisons[cache] = sess.run(self.poisoninputs)
        else:
            self.poisoninputs.load(self.cached_poisons[cache], sess)

    def restart_poison(self, sess):
        pert = np.random.uniform(-self.args.eps, self.args.eps, self.xbase.shape)
        self.poisoninputs.load(np.clip(self.xbase + pert, 0, 255), sess)
        # sess.run(self.broadcast_poisoninputs)

    def global_initialize(self, args, sess):
        sess.run(self.modified_global_initializer)
        if args.pretrain is not None:
            with open(args.pretrain, 'rb') as f: pretrain_weights = pickle.load(f)
            self.init_weights(sess, pretrain_weights)
            return pretrain_weights
    
