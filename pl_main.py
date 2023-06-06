# Import libraries
import torch, torchmetrics, wandb, timm, argparse, yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch import nn
from torch.nn import functional as F
from dataset import CustomDataloader
from transformations import get_tfs
from time import time

class LitModel(pl.LightningModule):
    
    """"
    
    This class gets several arguments and returns a model for training.
    
    Parameters:
    
        input_shape  - shape of input to the model, tuple -> int;
        model_name   - name of the model from timm library, str;
        num_classes  - number of classes to be outputed from the model, int;
        lr           - learning rate value, float.
    
    """
    
    def __init__(self, input_shape, model_name, num_classes, lr = 2e-4):
        super().__init__()
        
        # Log hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        # Evaluation metric
        self.accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = num_classes)
        # Get model to be trained
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
        self.train_times, self.validation_times = [], []

    # Get optimizere to update trainable parameters
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
        
    # Feed forward of the model
    def forward(self, inp): return self.model(inp)
    
    def on_train_epoch_start(self): self.train_start_time = time()
    
    def on_train_epoch_end(self): self.train_elapsed_time = time() - self.train_start_time; self.train_times.append(self.train_elapsed_time); self.log("train_time", self.train_elapsed_time, prog_bar = True)
        
    def training_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        x, y = batch
        # Get logits
        logits = self(x)
        # Compute loss        
        loss = F.cross_entropy(logits, y)
        # Get indices of the logits with max value
        preds = torch.argmax(logits, dim = 1)
        # Compute accuracy score
        acc = self.accuracy(preds, y)
        # Logs
        self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True); self.log("train_acc", acc, on_step = False, on_epoch = True, logger = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        x, y = batch
        # Get logits
        logits = self(x)
        # Compute loss
        loss = F.cross_entropy(logits, y)
        # Get indices of the logits with max value
        preds = torch.argmax(logits, dim = 1)
        # Compute accuracy score
        acc = self.accuracy(preds, y)
        # Log
        self.log("validation_loss", loss, prog_bar = True); self.log("validation_acc", acc, prog_bar = True)
        
        return loss
    
    def on_validation_epoch_start(self): self.validation_start_time = time()
    
    def on_validation_epoch_end(self): self.validation_elapsed_time = time() - self.validation_start_time; self.validation_times.append(self.validation_elapsed_time); self.log("validation_time", self.validation_elapsed_time, prog_bar = True)
    
    def get_stats(self): return self.train_times, self.validation_times
    
class ImagePredictionLogger(Callback):
    
    def __init__(self, val_samples, cls_names=None, num_samples=4):
        super().__init__()
        self.num_samples, self.cls_names = num_samples, cls_names
        self.val_imgs, self.val_labels = val_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        if self.cls_names != None:
            trainer.logger.experiment.log({
                "Sample Validation Prediction Results":[wandb.Image(x, caption=f"Predicted class: {self.cls_names[pred]}, Ground truth class: {self.cls_names[y]}") 
                               for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                     preds[:self.num_samples], 
                                                     val_labels[:self.num_samples])]
                })
        
def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments    
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    tfs = get_tfs(args.dataset_name)
    dl = CustomDataloader(root = args.root, transformations = tfs, bs = args.batch_size)
    # tr_dl, val_dl, test_dl = dl.get_dls()
    # torch.save(tr_dl, "saved_dls/tr_dl")
    # torch.save(val_dl, "saved_dls/val_dl")
    # torch.save(test_dl, "saved_dls/test_dl")
    
    tr_dl, val_dl, test_dl = torch.load("saved_dls/tr_dl"), torch.load("saved_dls/val_dl"), torch.load("saved_dls/test_dl")
    cls_names, n_cls = dl.get_info()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(val_dl))
    val_imgs, val_labels = val_samples[0], val_samples[1]

    # model = LitModel(args.inp_im_size, args.model_name, num_classes) if args.dataset_name == 'custom' else LitModel((32, 32), args.model_name, num_classes)
    model = LitModel(args.inp_im_size, args.model_name, n_cls) 

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='comparison', job_type='train', name=f"pl_lightning_multi_gpu_{args.devices}")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs = args.epochs, accelerator="gpu", devices = args.devices, strategy = "ddp", logger = wandb_logger,
                         callbacks = [EarlyStopping(monitor = 'validation_acc', mode = 'max', patience=20), ImagePredictionLogger(val_samples, cls_names),
                                      ModelCheckpoint(monitor = 'validation_loss', dirpath = args.save_model_path, filename = f'{args.model_name}_best')])

    
    start_time = time()
    trainer.fit(model, tr_dl, val_dl)
    train_times, valid_times = model.get_stats()
    torch.save(train_times, f"{args.stats_dir}/pl_train_times_{args.devices}_gpu")
    torch.save(valid_times[1:], f"{args.stats_dir}/pl_valid_times_{args.devices}_gpu")

    # Close wandb run
    wandb.finish()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = 'Image Classification Training Arguments')
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = 'dataset', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 256, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-dn", "--dataset_name", type = str, default = 'custom', help = "Dataset name for training")
    parser.add_argument("-mn", "--model_name", type = str, default = 'efficientnet_b0', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vit_base_patch16_224', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vgg16_bn', help = "Model name for backbone")
    parser.add_argument("-d", "--devices", type = int, default = 4, help = "Number of GPUs for training")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 20, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sd", "--stats_dir", type = str, default = "stats", help = "Path to dir to save train statistics")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)