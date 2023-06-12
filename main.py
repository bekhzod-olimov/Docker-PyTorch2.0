# Import Libraries
import argparse, torch, yaml, os, timm
from transformations import get_tfs
from dataset import CustomDataloader
from train import train
from utils import EarlyStopping

def run(args):
    
    """
    This function gets parsed arguments and trains the model.
    
    Arguments:
        
        args - parsed arguments, Parser object;
    
    """
    
    # Print Train Arguments    
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}")
    
    # Set Float Computation Precision
    torch.set_float32_matmul_precision("high")
    
    # Get Transformations
    tfs = get_tfs(args.ds_name)
    
    dl = CustomDataloader(root = args.root, transformations = tfs, bs = args.batch_size)
    # tr_dl, val_dl, test_dl = dl.get_dls()
    # torch.save(tr_dl, "saved_dls/tr_dl")
    # torch.save(val_dl, "saved_dls/val_dl")
    # torch.save(test_dl, "saved_dls/test_dl")
    
    tr_dl, val_dl, test_dl = torch.load("saved_dls/tr_dl"), torch.load("saved_dls/val_dl"), torch.load("saved_dls/test_dl")
    class_names, n_cls = dl.get_info()
    
    # Get Train Model
    model = timm.create_model(args.model_name, pretrained = True, num_classes = n_cls)
    if args.type == "2.0": model = torch.compile(model)
    
    # Get Training Details
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate)
    metric_to_track = "acc"
    early_stopping = EarlyStopping(metric_to_track = metric_to_track, patience = 20, threshold = 0.01)
     
    # Set Initial Best Accuracy    
    best_accuracy = 0.
    
    # Train Model
    train(model, tr_dl, val_dl, n_cls, criterion, optimizer, args.device, args.epochs, best_accuracy, args.ds_name, args.stats_dir, args.save_path, args.type, early_stopping, metric_to_track)   
    
if __name__ == "__main__":
    
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description = "PyTorch Model Training Arguments")
    parser.add_argument("-r", "--root", type = str, default = "dataset", help = "Dataset path")
    parser.add_argument("-t", "--type", type = str, default = "1.0", help = "PyTorch version")
    parser.add_argument("-bs", "--batch_size", type = int, default = 256, help = "Batch size")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:1', help = "GPU device number")
    parser.add_argument("-mn", "--model_name", type = str, default = 'efficientnet_b0', help = "Model name for training")
    parser.add_argument("-dn", "--ds_name", type = str, default = 'custom', help = "Dataset name for training")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 20, help = "Number of epochs")
    parser.add_argument("-sp", "--save_path", type = str, default = "saved_models", help = "Path to dir to save trained models")
    parser.add_argument("-sd", "--stats_dir", type = str, default = "stats", help = "Path to dir to save train statistics")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code based on the parsed arguments
    run(args) 
