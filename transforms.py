import torchvision.transforms as tfs

def get_tfs(ds_name):
    
    if ds_name == "cifar10" or ds_name == "cifar100":
        
        transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        return transform
    
    elif ds_name == "mnist":
        
        transform = tfs.Compose([tfs.ToTensor(), tfs.Grayscale(num_output_channels=3), tfs.Normalize((0.5), (0.5))])
        
        return transform