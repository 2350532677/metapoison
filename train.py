print('loading modules')
import os
from comet_ml import Experiment, ExistingExperiment, API
import tensorflow as tf
import argparse
from meta import Meta
from data import *
from utils import *
import pickle
import json
from time import time, sleep
from mpi4py import MPI

# initialize mpi
mpi = MPI.COMM_WORLD
nmeta = mpi.Get_size()
rank = mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

parser = argparse.ArgumentParser()
# logistics
parser.add_argument('-gpu', default=None, type=int, nargs='+')
parser.add_argument('-horovod', action='store_true')
parser.add_argument('-tf111', action='store_true')
parser.add_argument('-tag', default=None, type=str) # sgd or mom (momentum)
# parser.add_argument('-pretrain', default='log/no_aug-3/weights-epoch_190.pkl', type=str)
parser.add_argument('-pretrain', default=None, type=str)
parser.add_argument('-bit64', action='store_true')
parser.add_argument('-randomize', action='store_true')
parser.add_argument('-nocomet', action='store_true')
parser.add_argument('-weightset', default='standard-debug', type=str)
# threat model
parser.add_argument('-batchsize', default=125, type=int)
parser.add_argument('-nbatch', default=400, type=int)
parser.add_argument('-npoison', default=2000, type=int)
parser.add_argument('-ntarget', default=1, type=int)
# types of attacks
parser.add_argument('-multiclasspoison', action='store_true')
parser.add_argument('-targetclass', default=3, type=int)
parser.add_argument('-poisonclass', default=0, type=int)
parser.add_argument('-ytargetadv', default=None, type=int) # if none then use base class
# runtime/memory budget
parser.add_argument('-nvictimepoch', default=60, type=int)
# networks
parser.add_argument('-lrnrate', default=.1, type=float)
parser.add_argument('-warmupperiod', default=5, type=int)
parser.add_argument('-optimizer', default='sgd', type=str) # sgd or mom (momentum)
parser.add_argument('-weightdecay', action='store_true')
parser.add_argument('-augment', action='store_true')
parser.add_argument('-stagger', default=1, type=int)
parser.add_argument('-net', default='ResNet', type=str)
parser.add_argument('-droprate', default=0.1, type=float)
# unused, but needed for proper meta construction
parser.add_argument('-eps', default=16, type=float)
parser.add_argument('-nadapt', default=1, type=int)
parser.add_argument('-reducemethod', default='average', type=str) # softmax, average
parser.add_argument('-objective', default='xent', type=str) # cw, xent
args = parser.parse_args()
api = API()
if rank == 0:
    experiment = Experiment(parse_args=False, project_name=f'weightset-{args.weightset}', auto_param_logging=False, auto_metric_logging=False)
    experiment.log_parameters(vars(args))
    experiment.add_tag(args.tag)
args.gpu = set_available_gpus(args)

def victim():
    print('==> begin vanilla train')
    # begin training victim
    for epoch in range(args.nvictimepoch):
        tic = time()
        
        # log weights before each epoch to comet
        file = str(time()).replace('.', '')
        with open(file, 'wb') as f:
            pickle.dump(sess.run(meta.weights0), f)
        experiment.log_asset(file, file_name=f'weights0-{epoch}', step=epoch)
        os.remove(file)

        # start train epoch
        lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod)
        resLs, resVs = [], []
        for victimfeed in feeddict_generator(xtrain, ytrain, lrnrate, meta, args, victim=True):
            _, resL, = sess.run([meta.trainop, meta.resultL,], victimfeed)
            resLs.append(resL)
        for _, validfeed, _ in feeddict_generator(xvalid, yvalid, lrnrate, meta, args, valid=True):
            resV, = sess.run([meta.resultV,], validfeed)
            resVs.append(resV)
        # resL = avg_n_dicts(resLs)
        resV = avg_n_dicts(resVs)
        if not epoch % 1:
            experiment.log_metrics(resL, step=epoch)
            experiment.log_metrics(resV, step=epoch)
            print(' | '.join(['trainepoch {}'.format(epoch)] +
                             ['elapsed {}'.format(round(time() - tic, 4))] +
                             ['{} {}'.format(key, round(val, 2)) for key, val in resL.items()] +
                             ['{} {}'.format(key, round(val, 2)) for key, val in resV.items()]))
            
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
