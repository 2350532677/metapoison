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
from random import choice
from time import time, sleep
from mpi4py import MPI
import warnings
import multiprocessing  # Just for threadcounting in rank0
from subprocess import Popen, STDOUT, PIPE
import socket

# initialize mpi
mpi = MPI.COMM_WORLD
nproc = mpi.Get_size()
rank = mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

parser = argparse.ArgumentParser()
# logistics
parser.add_argument('-gpu', default=None, type=int, nargs='+')
parser.add_argument('-nocomet', action='store_true')
parser.add_argument('-runvictim', action='store_true')
parser.add_argument('-horovod', action='store_true')
parser.add_argument('-tf111', action='store_true')
parser.add_argument('-tag', default=None, type=str)
parser.add_argument('-job', default=None, type=str)
# parser.add_argument('-pretrain', default='log/no_aug-3/weights-epoch_190.pkl', type=str)
parser.add_argument('-pretrain', default=None, type=str)
parser.add_argument('-usestaggercache', action='store_true')
parser.add_argument('-bit64', action='store_true')
# threat model
parser.add_argument('-batchsize', default=125, type=int)
parser.add_argument('-nbatch', default=40, type=int)
parser.add_argument('-npoison', default=200, type=int)
parser.add_argument('-ntarget', default=1, type=int)
parser.add_argument('-eps', default=16, type=float)
# types of attacks
parser.add_argument('-multiclasspoison', action='store_true')
parser.add_argument('-targetclass', default=3, type=int)
parser.add_argument('-poisonclass', default=0, type=int)
parser.add_argument('-ytargetadv', default=None, type=int) # if none then use base class
# runtime/memory budget
parser.add_argument('-ncraftstep', default=30, type=int)
parser.add_argument('-nadapt', default=1, type=int)
parser.add_argument('-victimperiod', default=1, type=int)
parser.add_argument('-nreplay', default=1, type=int)
# networks
parser.add_argument('-lrnrate', default=.1, type=float)
parser.add_argument('-warmupperiod', default=5, type=int)
parser.add_argument('-optimizer', default='sgd', type=str) # sgd or mom (momentum)
parser.add_argument('-weightdecay', action='store_true')
parser.add_argument('-augment', action='store_true')
parser.add_argument('-stagger', default=1, type=int)
parser.add_argument('-net', default='ResNet', type=str)
parser.add_argument('-droprate', default=0.1, type=float)
parser.add_argument('-weightset', default='standard', type=str)
# poison optimization
parser.add_argument('-craftrate', default=200, type=float)
parser.add_argument('-patience', default=300, type=int)
parser.add_argument('-restartperiod', default=2000, type=int)
parser.add_argument('-reducemethod', default='average', type=str)  # softmax, average
parser.add_argument('-objective', default='xent', type=str)  # cw, xent
parser.add_argument('-trajectory', default='clean', type=str)  # clean or poison
args = parser.parse_args()

# Node info, especially important to check if the right ressources were allocated by SLURM
args.gpu = set_available_gpus(args)
ncpu = multiprocessing.cpu_count()
print('==> Rank {}/{}, localrank {}, host {}, GPU {}/{}, nCPUs {}'.format(rank, nproc,
      localrank, socket.gethostname(), localrank % len(args.gpu), len(args.gpu), ncpu))

# comet initialization
api = API()
weightapi = API()
experiment = Dummy()
if rank == 0 and not args.nocomet:
    experiment = Experiment(project_name='metapoison2', auto_param_logging=False, auto_metric_logging=False)
    experiment.add_tag(args.tag)
    experiment.log_parameters(vars(args))
    experiment.log_parameter('nproc', nproc)
    experiment.log_parameter('nmeta', args.nreplay * nproc)
    print(experiment.get_key())


def craft():
    def increment_or_reset_epoch():
        # restore weights to initial (with coldstart) after trainstep reaches k
        if Meta.epochstate == args.stagger * nproc:
            meta.init_weights(sess, pretrain_weights)
            Meta.epochstate = 0
        Meta.epochstate += 1

    def craftrate_schedule_gen():
        # craftrate scheduler
        bestval, bestidx = 1e32, 0
        craftrate = args.craftrate
        yield 0
        while True:
            # if resM['cwT'] < bestval - .01:
            #     bestval, bestidx = resM['cwT'], craftstep
            # elif craftstep - bestidx > args.patience:
            #     craftrate *= .5
            #     bestidx = craftstep
            #     if craftrate < .5:
            #         craftrate = 50
            if craftstep == 20: craftrate *= .5
            if craftstep == 40: craftrate *= .5
            yield min(((craftstep + 1) / 5) ** 2, 1) * craftrate

    def comet_log_asset_poison():
        # log poisons for victim eval
        fname = str(time()).replace('.', '')
        with open(fname, 'wb') as f:
            pickle.dump(sess.run(meta.poisoninputs), f)
        experiment.log_asset(fname, file_name='poisoninputs-{}'.format(craftstep), step=craftstep)
        os.remove(fname)

    def log_epoch_results(resM, resL, craftstep):
        # logging
        resL.update(dict(epochstate=Meta.epochstate, craftrate=craftrate))
        resMgather = mpi.gather(resM, root=0)
        resLgather = mpi.gather(resL, root=0)
        if rank == 0:
            resMseries.append(resMgather)
            resLseries.append(resLgather)
            resM, resL = avg_n_dicts(resMgather), avg_n_dicts(resLgather)
            experiment.log_metrics(resM, prefix='craft', step=craftstep)
            experiment.log_metrics(resL, prefix='craft', step=craftstep)
            if not craftstep % 1:
                print(' | '.join(['craftstep {}'.format(craftstep)] + ['elapsed {}'.format(round(time() - tic, 3))] +
                                 ['{} {}'.format(key, round(val, 2)) for key, val in resM.items()]))

    def comet_pull_weights_gen(api):
        allexpts = api.get(f'weightset-{args.weightset}')
        while True:
            tic = time()
            epoch = rank + replay * nproc
            while True:
                chosenexpt = choice(allexpts)
                assets = api.get_experiment_asset_list(chosenexpt.key)
                assets = [asset for asset in assets if asset['step'] == epoch]
                if len(assets) > 0: break
            asset = choice(assets)
            epoch = asset['step']
            weights0 = pickle.loads(api.get_experiment_asset(chosenexpt.key, asset['assetId']))
            print(f'rank {rank} weight {asset["assetId"]} from epoch {epoch} pulled from expt {chosenexpt.key} in {time() - tic} sec')
            yield weights0, epoch

    print('==> begin crafting poisons on rank {}'.format(rank))
    if rank == 0:
        resMseries, resLseries = [], []
        if args.runvictim:
            popen_args = dict(shell=True, env=os.environ, universal_newlines=True, encoding='utf-8', stdout=PIPE, stderr=STDOUT)
            command = 'horovodrun -np {} -H localhost:{} python victim.py {}'.format(len(args.gpu), len(args.gpu), experiment.get_key())
            victimproc = Popen(command, **popen_args)
    cr_schedule = craftrate_schedule_gen()
    comet_pull_weights = comet_pull_weights_gen(weightapi)
    for craftstep in range(args.ncraftstep):
        # auxiliary tasks
        tic = time()
        if not craftstep % args.victimperiod and rank == 0: comet_log_asset_poison()
        if not craftstep % args.restartperiod: meta.restart_poison(sess)
        # increment_or_reset_epoch()
        craftrate = next(cr_schedule)

        for replay in range(args.nreplay):
            # load new weights into network
            if rank == 0: print(f'replay {replay}')
            weights0, Meta.epochstate = next(comet_pull_weights)
            lrnrate = lr_schedule(args.lrnrate, Meta.epochstate, args.warmupperiod)
            meta.load_weights(sess, weights0)

            # iterate through all batches in epoch
            resMs = []
            for craftfeed, trainfeed, hasPoison in feeddict_generator(xtrain, ytrain, lrnrate, meta, args):
                resM = None
                if args.trajectory == 'clean':
                    if hasPoison: _, resM, = sess.run([meta.accumop, meta.resultM, ], craftfeed)
                    _, resL, = sess.run([meta.trainop, meta.resultL, ], trainfeed)
                elif args.trajectory == 'poison':
                    _, _, resM, resL, = sess.run([meta.accumop, meta.trainop, meta.resultM, meta.resultL, ], craftfeed)
                resMs.extend([] if resM is None else [resM])
        metagrad_accum = sess.run(meta.metagrad_accum)  # extract metagradients
        avg_metagrads = np.zeros_like(metagrad_accum)
        mpi.Allreduce(metagrad_accum, avg_metagrads, op=MPI.SUM) # use mpi rather than horovod to reduce
        avg_metagrads = avg_metagrads / (nproc * args.nreplay)
        sess.run([meta.craftop,], {meta.avg_metagrads: avg_metagrads, meta.craftrate: craftrate})
        resM = avg_n_dicts(resMs)
        log_epoch_results(resM, resL, craftstep)
    # process cleanup tasks
    if rank == 0:
        experiment.log_other('exitflag', True)
        plot_dict_series(resMseries, prefix='craft-indiv', experiment=experiment, step=0)
        plot_dict_series(resLseries, prefix='craft-indiv', experiment=experiment, step=0)
        if args.runvictim: victimproc.wait()

if __name__ == '__main__':

    # load data and build graph
    print('==> loading data on rank {}'.format(rank))
    xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytarget_adv = load_and_apportion_data(mpi, args)
    print('==> building graph on rank {}'.format(rank))
    meta = Meta(args, xbase, ybase, xtarget, ytarget_adv)

    # start tf session and initialize variables
    print('==> initializing tf session on rank {}'.format(rank))
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(localrank % len(args.gpu)))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    pretrain_weights = meta.global_initialize(args, sess)
    sess.graph.finalize()

    # begin
    craft()
