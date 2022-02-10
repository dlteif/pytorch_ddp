import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np
from model import _CNN
import wandb
from numpy import random

from utils import (load_state_dict, save_checkpoint, adjust_learning_rate, AverageMeter, accuracy, get_world_rank, get_world_size, get_local_rank, initialize_dist, get_cuda_device, allreduce_tensor)

from dataloader import prepare_dataloaders

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--wandb', action='store_true', help='for wandb logging')
    parser.add_argument('--description', type=str, help='wandb experiment name suffix')
    parser.add_argument('--job_id', type=str, help='wandb run name')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--dist', default=False, action='store_true', help='Use distributed training (default: no)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    parser.add_argument('--arch', type=str, default='CNN', help='Model architecture')
    parser.add_argument('--dataset', type=str, default='UNION', help='Dataset to train on')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--balanced', action='store_true', help='Balanced sampling')
    parser.add_argument('--loss', type=str, default='CE', help='Loss Criterion')

    
    parser.add_argument('--epochs', metavar='N', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--start_epoch', metavar='N', type=int, default=None, help='Starting epoch')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--num_workers', type=float, default=2, help='Number of workers for data loader')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    
    parser.add_argument('--eval', action='store_true', help='Evaluate model on test set')
    parser.add_argument('--resume', type=str, help='Path of the checkpoint to resume at')
    
    return parser


def train(model, device, train_loader, criterion, optimizer, epoch, start=0, frequency=3, _lambda=0, bg_loader=None, strategy='weight', rank=None):
    global output 
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
        
    with torch.autograd.set_detect_anomaly(True):
        for idx, (inputs, filenames, labels) in enumerate(tqdm(train_loader)):
            print('Batch ', idx)
            print(labels.size(), inputs.size())
            inputs = inputs.to(get_cuda_device())
            labels = labels.to(get_cuda_device())

            logits = model(inputs)
            print('logits: ', logits.size())
            probs = F.softmax(logits, dim=1)
            pred_probs, pred_labels = probs.sort(dim=1, descending=True)
            print('pred_probs: ', pred_probs.size(), ', pred_labels: ', pred_labels.size())

            loss = criterion(probs, labels)  

            prec1 = accuracy(probs, labels)[0]

            if args.dist:
                # Need to allreduce.
                # Could maybe get away with reduce, but this ensures all ranks
                # have the results.
                reduced_loss = allreduce_tensor(loss.data)
                reduced_prec1 = allreduce_tensor(prec1)
                losses.update(reduced_loss.item(), inputs.size(0))
                top1.update(reduced_prec1.item(), inputs.size(0))
            else:
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
    print(f'Train accuracy: {top1.avg}%, Train loss: {losses.avg}')

    if args.dist:
        torch.distributed.barrier()
    torch.cuda.synchronize()

    return top1.avg, losses.avg


def validate(model, device, loader, criterion, epoch=0, rank=None):
    global output
    model.eval()
    correct = 0.0
    total = 0.0
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    confusion_matrices = []
    correct_cls = [0.0,0.0]
    total_cls = [0.0,0.0]
    
    for idx, (inputs, filenames, labels) in enumerate(tqdm(loader)):
        print(labels.size())
        print(inputs.size())
        inputs = inputs.to(rank)
        labels = labels.to(rank)
        logits = model(inputs)
        print('logits: ', logits.size())
        print(logits)
        probs = F.softmax(logits, dim=1)
        pred_probs, pred_labels = probs.sort(dim=1, descending=True)
        loss = criterion(logits, labels)
        
        prec1 = accuracy(probs, labels)[0]
        
        if args.dist:
            # Need to allreduce.
            # Could maybe get away with reduce, but this ensures all ranks
            # have the results.
            reduced_loss = allreduce_tensor(loss.data)
            reduced_prec1 = allreduce_tensor(prec1)
            losses.update(reduced_loss.item(), inputs.size(0))
            top1.update(reduced_prec1.item(), inputs.size(0))
        else:
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
         
    
    print(f'Eval accuracy: {top1.avg}%, Eval loss: {losses.avg}')

    return top1.avg, losses.avg


def main(world_size, args, use_cuda):
    if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True
    if args.dist:
        initialize_dist(f'./init_{args.job_id}')
        
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    
    checkpoint_path = f'checkpoint_dir/{args.job_id}'  
    if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
    best_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0
    seed, repe_time = 1000, 5
    device = torch.device('cuda:0')
    if args.arch == 'CNN':
        net = _CNN(fil_num=20, drop_rate=0.5)
    else:
        pass


    if args.resume:
        ckpt = torch.load(args.resume,map_location=device)
        try:
            load_state_dict(net, ckpt)
        except:
            load_state_dict(net, ckpt['state_dict'])
            best_epoch = ckpt['epoch']
            best_acc = ckpt['accuracy']
            print(f'epoch: {best_epoch}, best acc: {best_acc}')
        
        print(f"loaded checkpoint at {args.resume}")
        
    
    rank = get_local_rank() if args.dist else None

    train_data, train_loader, val_data, val_loader, test_data, test_loader = prepare_dataloaders(args.dataset, args.data_dir, seed, batch_size=args.batch_size, num_workers=args.num_workers, balanced=args.balanced, world_size=world_size, rank=rank)
    
    if args.dist:
        model = DDP(net.to(get_cuda_device()), 
                    device_ids=[get_local_rank()],
                    output_device=get_local_rank(),
                    find_unused_parameters=True)
    else:
        model = net.to(get_cuda_device())

    
    if args.loss == 'CE':
        if args.balanced:
            weights, counts = train_data.get_sample_weights()
            print('counts: ', counts)
            if not isinstance(train_data.Label_list, list):
                count = torch.numel(train_data.Label_list)
            else:
                count = float(len(train_data.Label_list))
            print('class weights: ', count /torch.Tensor(counts))
            criterion = nn.CrossEntropyLoss(weight=count/torch.Tensor(counts)).to(get_cuda_device())
        else:
            criterion = nn.CrossEntropyLoss().to(get_cuda_device())
    else:
        pass

    
    if not args.eval:
        
        if not args.start_epoch is None:
            best_epoch = args.start_epoch
            
        for epoch in tqdm(range(best_epoch+1, args.epochs)):
            print(f'Epoch {epoch}')
            args.learning_rate = adjust_learning_rate(args.learning_rate, epoch, args.epochs)
            if args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
            elif args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            if args.dist:
                # IMPORTANT!
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)
                
            train_acc, train_loss = train(model, device, train_loader, criterion, optimizer, epoch=epoch, rank=get_local_rank())
            torch.cuda.synchronize()
            val_acc, val_loss = validate(model, device, val_loader, criterion, epoch=epoch, rank=get_local_rank())
            torch.cuda.synchronize()


            save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc,
                    'loss': val_loss,
            }, val_acc > best_acc, checkpoint_path, 'cnn')

            best_acc = max(best_acc, val_acc)

            log_dict = {
                'lr': args.learning_rate,
                'val_loss': val_loss,
                'best_accuracy': best_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
            }

            if args.wandb:
                wandb.log(log_dict, step=epoch)
    
    print('-----------------------------------------------------')
    print('                       Test                          ')
    print('-----------------------------------------------------')
    test_acc, test_loss  = validate(model, device, test_loader, criterion, rank=get_local_rank())
    if args.wandb:
        wandb.run.summary[f"test_accuracy"] = test_acc
        wandb.run.summary[f"test_loss"] = test_loss



if __name__ == '__main__':
    # global processes
    parser = get_parser()
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='PyTorch DDP', name=args.job_id)
        wandb.config.update({
             k: v for k, v in vars(args).items() if (isinstance(v, str) or isinstance(v, int) or isinstance(v, float))
            }, allow_val_change=True)

    use_cuda = args.gpus and torch.cuda.is_available()
    world_size = args.gpus
    out_str = str(args)
    if get_world_rank() == 0:
        print(out_str)
        print("We have available ", torch.cuda.device_count(), "GPUs! but using ",world_size," GPUs")
    
        
    main(world_size, args, use_cuda=use_cuda)