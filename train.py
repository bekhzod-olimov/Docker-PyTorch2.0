import timm, torch
from tqdm import tqdm
from time import time

def saveModel(ds_name, model):
    
    """
    
    Gets dataset name along with a model and saves it as the best model.
    
    Arguments:
    ds_name - dataset name;
    model - a trained model.
    
    """
    
    # Set the path to save the model
    path = f"best_models/{ds_name}_best_model.pth"
    
    # Save the model state_dictionary
    torch.save(model.state_dict(), path)

def validation(model, val_dl, device, ds_name):
    
    """
    
    Gets a model, validation dataloader, device type, and dataset name 
    performs one validation step and returns accuracy. 
    
    Arguments:
    model - a trained model;
    val_dl - validation dataloader;
    device - gpu type;
    ds_name - dataset name.

    """
    
    # Switch to evaluation model
    model.eval()
    
    # Set initial accuracy and total number of samples
    accuracy, total = 0, 0

    # Turn off gradient computation
    with torch.no_grad():
        
        # Go through the validation dataloader
        for i, batch in tqdm(enumerate(val_dl)):

            # Get images and labels
            images, labels = batch
            
            # Create 3 channel input for MNIST dataset
            if ds_name == "mnist":
                images_copy = images
                images = torch.cat((images, images_copy), dim = 1)
                images = torch.cat((images, images_copy), dim = 1)
                
            # Move images and labels to gpu
            images, labels = images.to(device), labels.to(device)
            
            # Get model predictions
            outputs = model(images)
            
            # Get the prediction with the max value
            _, predicted = torch.max(outputs.data, 1)
            
            # Add batch size to the total number of samples
            total += labels.size(0)
            
            # Compute accuracy over the mini-batch
            accuracy += (predicted == labels).sum().item()
    
    # Compute accuracy over the whole dataloader
    accuracy = (100 * accuracy / total)
    
    return accuracy
    
def train(model, tr_dl, val_dl, num_classes, criterion, optimizer, device, epochs, best_accuracy, ds_name):
    
    """
    
    Gets a number of train arguments and trains the model
    for the pre-defined number of epochs.
    performs one validation step and returns accuracy. 
    
    Arguments:
    model - a trained model;
    val_dl - validation dataloader;
    device - gpu type;
    ds_name - dataset name.

    """

    # Define your execution device
    print(f"The model will be running on {device} device")
    model.to(device)
    train_times, valid_times, accs = [], [], []
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        
        train_tic = time()
        running_loss, running_acc = 0, 0
        
        for i, batch in tqdm(enumerate(tr_dl, 0)):
            
            # get the inputs
            images, labels = batch
            
            # get 3 channels for mnist dataset
            if ds_name == "mnist":
                images_copy = images
                images = torch.cat((images, images_copy), dim=1)
                images = torch.cat((images, images_copy), dim=1)
                
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()
            running_loss += loss.item()     # extract the loss value

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        train_times.append(time() - train_tic)
        print(f"Training of epoch {epoch+1} is completed in {time() - train_tic:.3f} secs!\n")
        valid_tic = time()
        accuracy = validation(model, val_dl, device, ds_name)
        valid_times.append(time() - valid_tic)
        accs.append(accuracy)
        print(f"Validation of epoch {epoch+1} is completed in {time() - valid_tic:.3f} secs!\n")
        
        print(f"For epoch {epoch+1} the validation accuracy over the whole validation set is {accuracy:.2f}%\n")
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(ds_name, model)
            best_accuracy = accuracy
            
    torch.save(train_times, f'times2.0/{ds_name}_train_times')
    torch.save(valid_times, f'times2.0/{ds_name}_valid_times')
    torch.save(accs, f'times2.0/{ds_name}_accs')      
