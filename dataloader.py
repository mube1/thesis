

import torch
from torchvision import transforms
import torchvision

def prepare_dataloader(root,cifar=10,num_workers=8,download=False, train_batch_size=32, eval_batch_size=32):


    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    if cifar==10:
        train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform) 
        # We will use test set for validation and test in this project.
        # Do not use test set for validation in practice!
        test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
    else:
        train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=train_transform) 
        # We will use test set for validation and test in this project.
        # Do not use test set for validation in practice!
        test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=test_transform)

    # train_sampler = torch.utils.data.RandomSampler(train_set)
    train_sampler = torch.utils.data.SequentialSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader