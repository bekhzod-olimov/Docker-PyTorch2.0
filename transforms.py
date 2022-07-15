# Import libraries
import torchvision.transforms as tfs

def get_tfs(ds_name):
    
    """
    
    This function gets dataset name and returns transforms.
    
    Parameter:
    
        ds_name - dataset name, str.

    Output:

        tfs     - transformations to be applied, transforms object.
    
    """
    
    return tfs.Compose([tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), tfs.ToTensor()]) if "cifar" in ds_name else tfs.Compose([tfs.Normalize((0.5), (0.5)), tfs.ToTensor()])
