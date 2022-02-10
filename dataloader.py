import os
import math
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import DistributedSampler
import torchvision.transforms as transforms
import torch.distributed as dist
import numpy as np
import random
import csv

def read_csv(filename, dataset):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        # print(your_list[0])
        filenames = [''.join(a[:2]) for a in your_list[1:]]
        labels = torch.LongTensor([int(a[6]) for a in your_list[1:]])
        
        return filenames, labels


class CNN_Data(Dataset):
    """
    csv files ./lookuptxt/*.csv contains MRI filenames along with demographic and diagnosis information 
    """
    def __init__(self, dataset, filename, seed=1000, transform=None, filter=False):
        random.seed(seed)
        self.Data_list, self.Label_list = read_csv(filename, dataset)
        self.dataset = dataset
        self.transform = transform
        
        print(len(self.Data_list), len(self.Label_list))
        print(self.Data_list[0])

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        # print(self.Data_list[idx])
        data = np.load(self.Data_list[idx]).astype(np.float32)
            
        data = np.expand_dims(data, axis=0)
        if self.transform:
            data = self.transform(data)
        #         return data, label
        return data, self.Data_list[idx], label

    def get_sample_weights(self, label_list=None, data_sources=False):
        print('def get_sample_weights(): ', len(self.Label_list))
#         print(self.Label_list[:,1].size())
        if label_list is None:
            label_list = self.Label_list
        if self.dataset in ['UNION', 'ADNI_NACC']:
            label_list = label_list[:,1] if data_sources else self.Label_list[:,0]

        if not isinstance(label_list, list):
            label_list = label_list.tolist()
          
        count = float(len(label_list))
        print('total count: ', count)
        print(sorted(list(set(label_list))))
            
        uniques = sorted(list(set(label_list)))
        print('uniques: ',  uniques)
        counts = [float(label_list.count(i)) for i in uniques]
        print('counts: ', len(counts))
        
        weights = [count / counts[i] for i in label_list]
        # print('weights: ', weights)
        return weights, counts

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle

    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.Label_list
        # print(targets.size())
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
#         self.dataset.Label_list = targets
        weights, counts = self.dataset.get_sample_weights(label_list=targets)
        weights = torch.Tensor(weights)
        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch            


def prepare_dataloaders(dataset, data_dir, seed, batch_size=20, num_workers=2, balanced=False, augmentation=False, val_batch_size=10, world_size=None, rank=None):
    if augmentation:
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(50)
                                             ])
    else:
        train_transform = None
        
    train_data = CNN_Data(dataset, os.path.join(data_dir, 'train.csv'), seed=seed, transform=train_transform, filter=filter)
    val_data = CNN_Data(dataset, os.path.join(data_dir, 'val.csv'), seed=seed)
    test_data  = CNN_Data(dataset, os.path.join(data_dir, 'test.csv'), seed=seed)

    if balanced:
        sample_weight, counts = train_data.get_sample_weights()
        if rank != None:
            train_sampler = DistributedWeightedSampler(train_data, num_replicas=world_size, rank=rank, replacement=True, shuffle=True)
        else:    
            train_sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    
    if rank != None:
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    
    return train_data, train_loader, val_data, val_loader, test_data, test_loader


    