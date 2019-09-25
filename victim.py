print('loading modules')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
from comet_ml import Experiment, API
import tensorflow as tf
import argparse
from meta import Meta
from data import *
from utils import *
import pickle
import json
from time import time, sleep
from mpi4py import MPI
from socket import gethostname
from collections import defaultdict

mpi = MPI.COMM_WORLD
nmeta = mpi.Get_size()
rank = mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

parser = argparse.ArgumentParser()
# required: experiment key where crafted poisons
parser.add_argument('key')
parser.add_argument('-craftsteps', default=None, type=int, nargs='+')
# logistics
parser.add_argument('-gpu', default=None, type=int, nargs='+')
parser.add_argument('-horovod', action='store_true')
parser.add_argument('-tf111', action='store_true')
parser.add_argument('-tag', default=None, type=str)
# parser.add_argument('-pretrain', default='log/no_aug-3/weights-epoch_190.pkl', type=str)
parser.add_argument('-pretrain', default=None, type=str)
parser.add_argument('-bit64', action='store_true')
parser.add_argument('-name', default='', type=str)
# threat model
parser.add_argument('-batchsize', default=125, type=int)
parser.add_argument('-nbatch', default=8, type=int)
parser.add_argument('-npoison', default=10, type=int)
parser.add_argument('-ntarget', default=1, type=int)
# types of attacks
parser.add_argument('-multiclasspoison', action='store_true')
parser.add_argument('-targetclass', default=3, type=int)
parser.add_argument('-poisonclass', default=0, type=int)
parser.add_argument('-ytargetadv', default=None, type=int) # if none then use base class
# runtime/memory budget
parser.add_argument('-nvictimepoch', default=50, type=int)
# networks
parser.add_argument('-lrnrate', default=.05, type=float)
parser.add_argument('-warmupperiod', default=5, type=int)
parser.add_argument('-optimizer', default='mom', type=str) # sgd or mom (momentum)
parser.add_argument('-weightdecay', action='store_true')
parser.add_argument('-augment', action='store_true')
parser.add_argument('-net', default='ConvNet', type=str)
parser.add_argument('-droprate', default=0.1, type=float)
# overwrite
parser.add_argument('-Xweightdecay', action='store_true')
parser.add_argument('-Xaugment', action='store_true')
parser.add_argument('-Xoptimizer', action='store_true')
parser.add_argument('-Xbatchsize', action='store_true')
parser.add_argument('-Xlrnrate', action='store_true')
parser.add_argument('-Xnet', action='store_true')
parser.add_argument('-neval', default=None, type=int)
# unused, but needed for proper meta construction
parser.add_argument('-eps', default=16, type=float)
parser.add_argument('-nadapt', default=1, type=int)
parser.add_argument('-reducemethod', default='average', type=str)  # softmax, average
parser.add_argument('-objective', default='xent', type=str)  # cw, xent
args = parser.parse_args()
api = API()
attrs = ['lrnrate', 'batchsize', 'nbatch', 'npoison', 'ntarget', 'pretrain', 'warmupperiod', 'optimizer', 'weightdecay', 'augment', 'net', 'multiclasspoison', 'targetclass', 'poisonclass', 'ytargetadv']
copy_to_args_from_experiment(args, args.key, api, attrs)
if args.tag == 'multi': args.multiclasspoison = True
if args.Xweightdecay: args.weightdecay = True
if args.Xaugment: args.augment = True
if args.Xbatchsize: args.batchsize = 250
if args.Xlrnrate: args.lrnrate = .2
if args.Xoptimizer: args.optimizer = 'sgd'
if args.Xnet:
    args.net = 'ConvNet'
    args.lrnrate = .05
print(args)
if rank == 0:
    experiment = Experiment(project_name='metapoison-victim', auto_param_logging=False, auto_metric_logging=False)
    experiment.log_parameters(vars(args))
    experiment.log_parameter('nmeta', nmeta)
    experiment.set_name(args.key)
    experiment.add_tag(args.tag)
args.gpu = set_available_gpus(args)
if args.name == '': args.name = args.net


def victim():
    def comet_pull_next_poison():
        # grab next poison from comet that hasn't been processed
        impatience = 0
        while not has_exitflag(args.key, api) or impatience < 5:  # patience before ending victim process
            sleep(1)
            print('searching for poisons to pull')
            assets = {asset['step']: asset['assetId'] for asset in api.get_experiment_asset_list(args.key)
                      if 'poisoninputs-' in asset['fileName']}
            logged = set(metric['step'] for metric in api.get_experiment_metrics_raw(args.key)
                         if metric['metricName'] == 'victim{}_accT0'.format(args.name))
            unlogged = set(assets.keys()) - logged - locallog
            if args.craftsteps is not None:
                unlogged = set(args.craftsteps).intersection(set(assets.keys())) - locallog
            if len(unlogged) > 0:
                craftstep = max(unlogged)
                locallog.add(craftstep)
                bytefile = api.get_experiment_asset(args.key, assets[craftstep])
                print('==> poisoninputs-{} pulled'.format(craftstep))
                return pickle.loads(bytefile), craftstep
            impatience += 1
        return None, None

    def comet_log_figure_poison():
        # log poison images onto comet
        npoison_to_display = 8
        for i in np.linspace(0, args.npoison - 1, npoison_to_display, dtype=int):
            imagesc(poisoninputs[i], title='poison-{}'.format(i), experiment=experiment, step=craftstep, scale=[0, 255])
            # imagesc(poisoninputs[i] - xbase[i], title='perturb-{}'.format(i), experiment=experiment, step=craftstep, scale=127.5)

    def comet_log_asset(asset, name, epoch=''):
        fname = str(time()).replace('.', '')
        with open(fname, 'wb') as f:
            pickle.dump(feats, f)
        experiment.log_asset(fname, file_name='{}-{}-{}'.format(name, craftstep, epoch), step=craftstep)
        os.remove(fname)

    print('==> begin victim train')
    locallog = set()
    totaleval = defaultdict(int)
    cnt = 0
    while args.neval is None or cnt < args.neval:
        cnt += 1
        if len(set(args.craftsteps) - locallog) == 0: locallog = set() # start over

        # pull poisons from comet
        poisoninputs, craftstep = None, None
        if rank == 0: poisoninputs, craftstep = comet_pull_next_poison()
        poisoninputs, craftstep = mpi.bcast((poisoninputs, craftstep), root=0)
        if poisoninputs is None:
            print('no more poisons to process')
            return  # kill victim run when theres no more poisons left to process
        if rank == 0: comet_log_figure_poison()
        meta.init_weights(sess, pretrain_weights)
        meta.poisoninputs.load(poisoninputs, sess)

        # begin training victim
        resLseries, resVseries = [], []
        for epoch in range(args.nvictimepoch):
            print('debug')
            tic = time()
            lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod)
            resLs, resVs, feats = [], [], defaultdict(list)
            for victimfeed in feeddict_generator(xtrain, ytrain, lrnrate, meta, args, victim=True):
                _, resL, feat = sess.run([meta.trainop, meta.resultL, meta.feat,], victimfeed)
                resLs.append(resL)
                appendfeats(feats, feat, victimfeed, ybase, ytarget)
            for _, validfeed, _ in feeddict_generator(xvalid, yvalid, lrnrate, meta, args, valid=True):
                resV, = sess.run([meta.resultV,], validfeed)
                resVs.append(resV)
            resL = avg_n_dicts(resLs)
            resL.update(dict(featdist=get_featdist(feats)))
            resV = avg_n_dicts(resVs)
            resLseries.append(resL)
            resVseries.append(resV)
            if rank == 0 and (not (epoch + 1) % 20 and epoch != 1): comet_log_asset(feats, 'feats', epoch)
            if not epoch % 1:
                if rank == 0: experiment.log_metrics(resL, prefix=f'craftstep-{craftstep}-{totaleval[craftstep]}', step=epoch)
                if rank == 0: experiment.log_metrics(resV, prefix=f'craftstep-{craftstep}-{totaleval[craftstep]}', step=epoch)
                if rank == 0: print(f'https://www.comet.ml/wronnyhuang/metapoison-victim/{experiment.get_key()}')
                if rank == 0: print(args)
                print(' | '.join(['{}-{} | {} | poisoninputs-{} | epoch {} | elapsed {}'.format(args.key[:8], args.name, gethostname(), craftstep, epoch, round(time() - tic, 3))] +
                                 ['{} {}'.format(key, round(val, 2)) for key, val in resLseries[-1].items()] +
                                 ['{} {}'.format(key, round(val, 2)) for key, val in resVseries[-1].items()]))

        # log end result and also individual training curves
        resLgather = mpi.gather(resLseries, root=0)
        resVgather = mpi.gather(resVseries, root=0)
        if rank == 0:
            totaleval[craftstep] += 1
            print(f'totaleval: {dict(totaleval)}')
            experiment.log_metric('totaleval', sum([t for t in totaleval.values()]), step=sum([t for t in totaleval.values()]))
            resLseries = transpose_list_of_lists(resLgather)
            resVseries = transpose_list_of_lists(resVgather)
            experiment.log_metrics(avg_n_dicts([avg_n_dicts(r) for r in resLseries[-15:]]),
                                   prefix='victim{}'.format(args.name), step=craftstep)
            experiment.log_metrics(avg_n_dicts([avg_n_dicts(r) for r in resVseries[-15:]]),
                                   prefix='victim{}'.format(args.name), step=craftstep)
            # comet_log_asset(resLseries, 'resLseries')
            # comet_log_asset(resVseries, 'resVseries')
            # plot_dict_series(resLseries, prefix='victim{}-indiv'.format(args.name),
            #                  experiment=experiment, step=craftstep)
            # plot_dict_series(resVseries, prefix='victim{}-indiv'.format(args.name),
            #                  experiment=experiment, step=craftstep)


if __name__ == '__main__':

    # load data and build graph
    print('==> loading data')
    xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytarget_adv = load_and_apportion_data(mpi, args)
    print('==> building graph')
    meta = Meta(args, xbase, ybase, xtarget, ytarget_adv)

    # start tf session and initialize variables
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(localrank % len(args.gpu)))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    pretrain_weights = meta.global_initialize(args, sess)
    sess.graph.finalize()

    # begin
    victim()
