import argparse, torch, yaml, os, timm
from dataset import get_dl
from transforms import get_tfs
import torchvision.models as models
from train import train

def run(args):
    
    model_name = args.model_name
    ds_name = args.ds_name
    epochs = args.epochs
    device = args.device
    bs = args.batch_size
    lr = args.learning_rate    
    
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nTraining Arguments:\n{argstr}")
    torch.set_float32_matmul_precision('high')
    
    # Get Data
    tfs = get_tfs(ds_name)
    tr_dl, val_dl, num_classes = get_dl(ds_name, tfs=tfs, bs=bs)
    
    # Get Model
    m = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = m
    # model = torch.compile(m)
    
    # Get Training Details
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
     
    # Train model    
    best_accuracy = 0.
    train(model, tr_dl, val_dl, num_classes, criterion, optimizer, device, epochs, best_accuracy, ds_name)   
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch 2.0 Model Training Arguments')
    parser.add_argument("-bs", "--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("-d", "--device", type=str, default='cuda:3', help="GPU device number")
    parser.add_argument("-mn", "--model_name", type=str, default='efficientnet_b0', help="Model name for training")
    parser.add_argument("-dn", "--ds_name", type=str, default='cifar10', help="Dataset name for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate value")
    parser.add_argument("-e", "--epochs", type=int, default=25, help="Number of epochs")
    args = parser.parse_args() 
    
    run(args) 
    
