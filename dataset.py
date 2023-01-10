import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def get_dl(ds_name, tfs, bs):
    
    """ 
    
    Gets dataset name, transformations, and batch size and returns train, test dataloaders along with number of classes.
    
    Arguments:
    ds_name - dataset name;
    tfs - transformations;
    bs - batch size. 
    
    """
    
    # Assertions for the dataset name
    assert ds_name == "cifar10" or ds_name == "cifar100" or ds_name == "mnist", "Please choose one of these datasets: mnist, cifar10, cifar100"
    
    # CIFAR10 dataset
    if ds_name == "cifar10":
        
        # Get trainset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
        
        # Get testset
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tfs)
        
        # Initialize test dataloader
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))
    
    # CIFAR100 dataset
    elif ds_name == "cifar100":
        
        # Get trainset
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
        
        # Get testset
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=tfs)
        
        # Initialize test dataloader
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))
    
    elif ds_name == "mnist":
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=tfs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tfs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))

    print(f"{ds_name} is loaded successfully!")
    print(f"{ds_name} has {num_classes} classes!")
    
    return trainloader, testloader, num_classes

        
        
    
    
    
    



