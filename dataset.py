# Import libraries
import torch, torchvision, os
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
torch.manual_seed(2023)

def get_dl(ds_name, tfs, bs):
    
    """ 
    
    This function gets dataset name, transformations, and batch size and returns train, test dataloaders along with number of classes.
    
    Parameters:
    
        ds_name        - dataset name, str;
        tfs            - transformations, torchvision transforms object;
        bs             - batch size, int. 
        
    Outputs:
    
        trainloader    - train dataloader, torch dataloader object;
        testloader     - test dataloader, torch dataloader object;
        num_classes    - number of classes in the dataset, int.
    
    """
    
    # Assertions for the dataset name
    assert ds_name == "cifar10" or ds_name == "cifar100" or ds_name == "mnist", "Please choose one of these datasets: mnist, cifar10, cifar100"
    
    # CIFAR10 dataset
    if ds_name == "cifar10":
        
        # Get trainset
        trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True)
        
        # Get testset
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = tfs)
        
        # Initialize test dataloader
        testloader = torch.utils.data.DataLoader(testset, batch_size = bs, shuffle = False)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))
    
    # CIFAR100 dataset
    elif ds_name == "cifar100":
        
        # Get trainset
        trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle = True)
        
        # Get testset
        testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = tfs)
        
        # Initialize test dataloader
        testloader = torch.utils.data.DataLoader(testset, batch_size = bs, shuffle = False)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))
    
    # MNIST dataset
    elif ds_name == "mnist":
        
        # Get trainset
        trainset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True)
        
        # Get testset
        testset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = tfs)
        
        # Initialize test dataloader
        testloader = torch.utils.data.DataLoader(testset, batch_size = bs, shuffle = False)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))

    print(f"{ds_name} is loaded successfully!")
    print(f"{ds_name} has {num_classes} classes!")
    
    return trainloader, testloader, num_classes

class CustomDataloader(nn.Module):
    def __init__(self, root, transformations, bs, im_files = [".jpg", ".png", ".jpeg"], data_split = [0.8, 0.1, 0.1]):
        super().__init__()
        
        assert sum(data_split) == 1, "Data split elements' sum must be exactly 1"
        self.im_files, self.bs = im_files, bs
        self.ds = ImageFolder(root = root, transform = transformations, is_valid_file = self.check_validity)
        
        self.total_ims = len(self.ds)
        tr_len = int(self.total_ims * data_split[0])
        val_len = int(self.total_ims * data_split[1])
        test_len = self.total_ims - (tr_len + val_len)
        self.tr_ds, self.val_ds, self.test_ds = random_split(dataset = self.ds, lengths = [tr_len, val_len, test_len])
        self.all_ds = {"train": self.tr_ds, "validation": self.val_ds, "test": self.test_ds}
        for idx, (key, value) in enumerate(self.all_ds.items()): print(f"There are {len(value)} images in the {key} dataset.")
        
    def __len__(self): return self.total_ims

    def check_validity(self, path):
        if os.path.splitext(path)[-1] in self.im_files: return True
        return False
    
    def get_dls(self): return [DataLoader(dataset = ds, batch_size = self.bs, shuffle = True, num_workers = 8) for ds in self.all_ds.values()]
    
    def get_info(self): return self.ds.classes, len(self.ds.classes)
        
# tfs = T.Compose([T.Resize((224,224)), T.ToTensor()])
# ddl = CustomDataloader(root = "dataset", transformations = tfs, bs = 64)
# tr_dl, val_dl, test_dl = ddl.get_dls()
# a, b = ddl.get_info()
