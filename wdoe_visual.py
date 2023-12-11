# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from models.wrn import WideResNet
import utils.utils_awp as awp
import utils.svhn_loader as svhn
from utils.display_results import  get_measures, print_measures
from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with DOE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/pretrained', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')

parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--gamma', type=int, default=1)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--begin_epoch', type=int, default=0)



args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True  # fire on all cylinders

print('warmup %d gamma %d alpha %.2f' % (args.warmup, args.gamma, args.alpha))

# mean and standard deviation of channels of CIFAR-10 images
if 'cifar' in args.dataset:
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
else: 
    mean= torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).tolist()
    std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).tolist()


train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10('../data/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
    cifar_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform) 
    num_classes = 10
else:
    train_data_in = dset.CIFAR100('../data/cifarpy', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform)
    cifar_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
    num_classes = 100
calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'
ood_data = dset.ImageFolder(root="../data/tiny-imagenet-200/train/", transform=trn.Compose([trn.Resize(32), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))

'''from utils.tinyimages_80mn_loader import TinyImages
ood_data = TinyImages(transform=trn.Compose([trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]), exclude_cifar = True)
'''
train_loader_in = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=False)
train_loader_out = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=False)

texture_data = dset.ImageFolder(root="../doe/data/dtd/images", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
places365_data = dset.ImageFolder(root="../data/places365_standard/", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
isun_data = dset.ImageFolder(root="../data/iSUN",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
svhn_data = svhn.SVHN(root='../data/svhn/', split="test",transform=trn.Compose( [trn.ToTensor(), trn.Normalize(mean, std)]), download=False)

texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
cifar_loader = torch.utils.data.DataLoader(cifar_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.cuda(), target.cuda()
            output = net(data)# .softmax(1)
            smax = to_np(output)
            _score.append(-np.max(smax, axis=1))
    if in_dist:
        return concat(_score).copy() 
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(ood_loader, in_score, num_to_avg=args.num_to_avg):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')
    return out_score

def train(epoch, diff):
    proxy = WideResNet(args.layers, num_classes, args.widen_factor, dropRate = 0).cuda()
    proxy_optim = torch.optim.SGD(proxy.parameters(), lr=1)
    net.train()
    loss_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_in.dataset))
    list_id_features = []
    list_ood_features = []
    list_wor_features = []
    list_org_loss = []
    list_wor_loss = []
    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        data, target = torch.cat((in_set[0], out_set[0]), 0), in_set[1]
        data, target = data.cuda(), target.cuda()

        id_logits, id_embs   = net.feature_list(data[:len(in_set[0])])
        ood_logits, ood_embs = net.feature_list(data[len(in_set[0]):])
        id_features, ood_features = id_embs[-1], ood_embs[-1]
        org_loss = - (ood_logits.mean(1) - torch.logsumexp(ood_logits, dim=1))
        
        org_embs = ood_features
        gamma =  torch.Tensor([1e-1,1e-2,1e-3,1e-4])[torch.randperm(4)][0]
        proxy.load_state_dict(net.state_dict())
        proxy.train()
        try: awp.add_into_weights(proxy, diff, coeff = - gamma)
        except: pass
        scale = torch.Tensor([1]).cuda().requires_grad_()
        logits, embs = proxy.feature_list(data[len(in_set[0]):])
        x = logits * scale
        l_reg = ((org_embs - embs[-1]) ** 2).mean(-1).mean()
        l_sur = - (x.mean(1) - torch.logsumexp(x, dim=1)).mean()
        reg_sur = torch.sum(torch.autograd.grad(l_sur, [scale], create_graph = True)[0] ** 2)
        proxy_optim.zero_grad()
        (reg_sur - args.gamma * l_reg).backward()
        torch.nn.utils.clip_grad_norm_(proxy.parameters(), 1)
        proxy_optim.step()
        try:    diff = awp.average_diff(diff, awp.diff_in_weights(net, proxy), beta = .6)
        except: diff = awp.diff_in_weights(net, proxy)
        awp.add_into_weights(net, diff, coeff = - gamma)
        ood_logits, ood_embs = net.feature_list(data[len(in_set[0]):])
        wor_features = ood_embs[-1]
        wor_loss = - (wor_features.mean(1) - torch.logsumexp(wor_features, dim=1))
        awp.add_into_weights(net, diff, coeff = gamma)

        if batch_idx > 100:
            list_id_features.append(  id_features.detach())
            list_ood_features.append(ood_features.detach())
            list_wor_features.append(wor_features.detach())
            list_org_loss.append(org_loss.detach())
            list_wor_loss.append(wor_loss.detach())
        if batch_idx == 110: break
    return list_id_features, list_ood_features, list_wor_features, list_org_loss, list_wor_loss

net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
model_path = 'woe_%s.pth' % args.dataset
net.load_state_dict(torch.load(model_path))
list_id_features, list_ood_features, list_wor_features, list_org_loss, list_wor_loss = train(0, None)

torch.save({
    'list_id_features': list_id_features,
    'list_ood_features': list_ood_features,
    'list_wor_features': list_wor_features,
    'list_org_loss': list_org_loss,
    'list_wor_loss': list_wor_loss,
}, 'features_%s.pth' % args.dataset)


net.eval()
in_score = get_ood_scores(test_loader, in_dist=True)
ood_scores = []
ood_scores.append(get_and_print_results(svhn_loader, in_score))
ood_scores.append(get_and_print_results(lsunc_loader, in_score))
ood_scores.append(get_and_print_results(isun_loader, in_score))
ood_scores.append(get_and_print_results(texture_loader, in_score))
ood_scores.append(get_and_print_results(places365_loader, in_score))


torch.save({
    'list_id_features': list_id_features,
    'list_ood_features': list_ood_features,
    'list_wor_features': list_wor_features,
    'list_org_loss': list_org_loss,
    'list_wor_loss': list_wor_loss,
    'id_scores': in_score,
    'ood_scores': ood_scores,
}, 'features_%s.pth' % args.dataset)

