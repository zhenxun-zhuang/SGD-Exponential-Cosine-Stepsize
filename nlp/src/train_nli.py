import sys
import os
import argparse
import numpy as np
import random
import mlogger
import torch
import torch.nn as nn

from hinge_loss import MultiClassHingeLoss, set_smoothing_enabled
from data_util import get_nli, get_batch, build_vocab
from models_nil import NLINet
from tqdm import tqdm

from optim import get_optimizer, DFW
from utils import adapt_grad_norm, setup_xp, accuracy


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--data_root", type=str, default='./datasets', help="data root")
parser.add_argument("--log_folder", type=str, default='./logs', help="folder to save logs")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--server", type=str, default='http://127.0.0.1')
parser.add_argument("--port", type=int, default=9003)
parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help="use of tqdm progress bars")
parser.set_defaults(tqdm=True)


# training
parser.add_argument("--validation", action='store_true', help="validation or test")
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--opt", type=str, default="sgd", help="choice of optimizer")
parser.add_argument("--eta", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--alpha", type=float, default=0.1, help="used in SGD decays")
parser.add_argument("--momentum", type=float, default=0, help="momentum")
parser.add_argument("--l2", type=float, default=0., help="l2-regularization")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument('--load-opt', default=None, help='data file with opt')

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--loss", type=str, default='svm', help="choice of loss function")
parser.add_argument("--smooth-svm", dest="smooth_svm", action='store_true', help="smoothness of SVM")
parser.set_defaults(smooth_svm=False)

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=None, help="seed")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
if params.seed is None:
    params.seed = np.random.randint(1e5)
print('Seed:\t {}'.format(params.seed))
random.seed(params.seed)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)


"""
DATA
"""
GLOVE_PATH = params.data_root + "/GloVe/glove.840B.300d.txt"

train, valid, test = get_nli(os.path.join(params.data_root, 'SNLI'))
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], GLOVE_PATH)

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

params.word_emb_dim = 300


"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)

nli_net = NLINet(config_nli_model)
print(nli_net)

# loss
if params.loss == 'svm':
    loss_fn = MultiClassHingeLoss()
else:
    loss_fn = nn.CrossEntropyLoss()

# cuda by default
nli_net.cuda()
loss_fn.cuda()

n_batches_per_epoch = np.ceil(len(train['s1']) / params.batch_size)
T_max = params.n_epochs * n_batches_per_epoch
optimizer = get_optimizer(params, nli_net.parameters(), T_max)

params.visdom = False
xp_name = '{}--{}--{}--eta-{}'.format(params.encoder_type, params.opt, params.loss, params.eta)
if params.smooth_svm:
    xp_name = xp_name + '--smooth'
if params.opt in ['SGD_Exp_Decay', 'SGD_1sqrt_Decay', 'SGD_1t_Decay', 'SGD_Stage_Decay']:
    xp_name = xp_name + ('--alpha-{}'.format(params.alpha))
xp_name += '--{}'.format(params.seed)
xp_name += '--validation' if params.validation else ''

params.log = True
params.outputdir = params.log_folder + '/saved/{}'.format(xp_name)
params.xp_name = params.log_folder + '/xp_results/{}'.format(xp_name)

if not os.path.exists(params.xp_name):
    os.makedirs(params.xp_name)
else:
    raise ValueError('Experiment already exists at {}'.format(params.xp_name))
xp = setup_xp(params, nli_net, optimizer)


"""
TRAIN
"""
def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))
    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]

    if epoch > 1 and params.opt == 'sgd':
        optimizer.param_groups[0]['lr'] *= params.decay
        optimizer.step_size = optimizer.param_groups[0]['lr']

    for metric in xp.train.metrics():
        metric.reset()

    for stidx in tqdm(range(0, len(s1), params.batch_size), disable=not params.tqdm,
                      desc='Train Epoch', leave=False):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word_vec)
        s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
        tgt_batch = torch.LongTensor(target[stidx:stidx + params.batch_size]).cuda()

        # model forward
        scores = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        with set_smoothing_enabled(params.smooth_svm):
            loss = loss_fn(scores, tgt_batch)

        # backward
        optimizer.zero_grad()
        loss.backward()
        if not isinstance(optimizer, DFW):
            adapt_grad_norm(nli_net, params.max_norm)
        # necessary information for the step-size of some optimizers -> provide closure
        optimizer.step(lambda: float(loss))

        # monitoring
        batch_size = scores.size(0)
        xp.train.acc.update(accuracy(scores, tgt_batch), weighting=batch_size)
        xp.train.loss.update(loss_fn(scores, tgt_batch), weighting=batch_size)
        xp.train.gamma.update(optimizer.gamma, weighting=batch_size)

    xp.train.eta.update(optimizer.eta)
    xp.train.reg.update(0.5 * params.l2 * xp.train.weight_norm.value ** 2)
    xp.train.obj.update(xp.train.reg.value + xp.train.loss.value)
    xp.train.timer.update()

    for metric in xp.train.metrics():
        metric.log(time=xp.epoch.value)


val_acc_best = -1e10

def evaluate(epoch, eval_type, final_eval=False):
    nli_net.eval()
    global val_acc_best, lr

    if eval_type == 'train':
        tag = 'Train'
        xp_group = xp.train
        s1, s2, target = train['s1'], train['s2'], train['label']
    elif eval_type == 'valid':
        tag = 'Valid'
        xp_group = xp.val
        s1, s2, target = valid['s1'], valid['s2'], valid['label']
    else:
        tag = 'Test '
        xp_group = xp.test
        s1, s2, target = test['s1'], test['s2'], test['label']

    for metric in xp_group.metrics():
        metric.reset()

    for i in tqdm(range(0, len(s1), params.batch_size), disable=not params.tqdm,
                  desc='{} Epoch'.format(tag.title()), leave=False):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
        tgt_batch = torch.LongTensor(target[i:i + params.batch_size]).cuda()

        # model forward
        scores = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        batch_size = scores.size(0)
        xp_group.acc.update(accuracy(scores, tgt_batch), weighting=batch_size)
        xp_group.loss.update(loss_fn(scores, tgt_batch), weighting=batch_size)

    xp_group.timer.update()

    print('Epoch: %d (%s)   Loss %g   Acc %g\n' % (epoch, tag, xp_group.loss.value, xp_group.acc.value))

    if tag == 'val':
        xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)

    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)

    eval_acc = xp_group.acc.value
    # save model
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net, os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if params.opt == 'sgd' :
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))


"""
Train model on Natural Language Inference task
"""
epoch = 1

with mlogger.stdout_to("{}/log.txt".format(params.xp_name)):
    while epoch <= params.n_epochs:
        xp.epoch.update(epoch)
        trainepoch(epoch)
        evaluate(epoch, 'train')
        evaluate(epoch, 'valid')
        epoch += 1

    # Run best model on test set.
    del nli_net
    nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))

    print('\nTEST : Epoch {0}'.format(epoch))
    evaluate(1e6, 'valid', True)
    if not params.validation:
        evaluate(0, 'test', True)

    # Save encoder instead of full model
    torch.save(nli_net.encoder,
               os.path.join(params.outputdir, params.outputmodelname + '.encoder'))
