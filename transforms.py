# Import libraries
import torchvision.transforms as tfs

def get_tfs(ds_name):
    
    """
    
    This function gets dataset name and returns transforms.
    
    Arguments:
    
        ds_name - dataset name.
    
    """
    
    # CIFAR10 and CIFAR100
    if ds_name == "cifar10" or ds_name == "cifar100":
        
        # Tensor and Standard Normalization
        transform = tfs.Compose([tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), tfs.ToTensor()])
        
        return transform
    
    # MNIST 
    elif ds_name == "mnist":
        
        # Tensor and Standard Normalization
        transform = tfs.Compose([tfs.Normalize((0.5), (0.5)), tfs.ToTensor()])
        
        return transform
