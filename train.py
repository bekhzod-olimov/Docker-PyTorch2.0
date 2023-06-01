# Import libraries
import timm, torch
from tqdm import tqdm
from time import time

def saveModel(ds_name, model):
    
    """
    
    This function gets dataset name along with a model and saves it as the best model.
    
    Parameters:
    
        ds_name     - dataset name, str;
        model       - a trained model, timm model object.
    
    """
    
    # Set the path to save the model
    path = f"best_models/{ds_name}_best_model.pth"
    
    # Save the model state_dictionary
    torch.save(model.state_dict(), path)

def validation(model, val_dl, device, ds_name):
    
    """
    
    This function gets several parameters and performs one validation step and returns accuracy. 
    
    Parameters:
    
        model        - a trained model, timm model object;
        val_dl       - validation dataloader, torch dataloader object;
        device       - gpu type, str;
        ds_name      - dataset name, str.
        
    Output:
    
        accuracy     - accuracy percentage on the validation dataloader, float.

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
                images = torch.cat((images, images_copy), dim = 1); images = torch.cat((images, images_copy), dim = 1)
                
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
    return 100 * accuracy / total
    
def train(model, tr_dl, val_dl, num_classes, criterion, optimizer, device, epochs, best_accuracy, ds_name):
    
    """
    
    This function gets several parameters, performs one train epoch and one validation epoch and returns accuracy. 
    
    Parameters:
    
        model            - a trained model, timm model object;
        tr_dl            - train dataloader, torch dataloader object;
        num_classes      - number of classes in the dataset, int;
        criterion        - loss function, torch object;
        optimizer        - optimizer to update the weights, torch optimizer object;
        device           - gpu type, str;
        epochs           - number of epochs to train the model, int;
        best_accuracy    - initial value for the best accuracy, float;
        ds_name          - dataset name, str.

    """

    # Define the gpu device and move the model to the gpu
    print(f"The model will be running on {device} device")
    model.to(device)
    
    # Initialize lists to track train metrics
    train_times, valid_times, accs = [], [], []
    
    # Start train process
    for epoch in range(epochs): 

        # Set start train time
        train_tic = time()
        
        # Set initial values for loss and accuracy
        running_loss, running_acc = 0, 0
        
        # Go through the train dataloader
        for i, batch in tqdm(enumerate(tr_dl, 0)):
            
            # Get images and labels
            images, labels = batch
            
            # Make 3 channel input for the MNIST dataset
            if ds_name == "mnist":
                
                images_copy = images
                images = torch.cat((images, images_copy), dim = 1)
                images = torch.cat((images, images_copy), dim = 1)
            
            # Move images and labels to gpu
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Get model predictions
            outputs = model(images)
            
            # Comput loss value
            loss = criterion(outputs, labels)
            
            # Backpropagation
            loss.backward()
            
            # Update the model weights
            optimizer.step()
            
            # Add mini-batch loss to the total loss
            running_loss += loss.item()

        # Add one-epoch train time to the list
        train_times.append(time() - train_tic)
        print(f"Training of epoch {epoch+1} is completed in {time() - train_tic:.3f} secs!\n")
        
        # Start validation process
        valid_tic = time()
        
        # Get validation accuracy
        accuracy = validation(model, val_dl, device, ds_name)
        
        # Add one-epoch validation time to the list
        valid_times.append(time() - valid_tic)
        
        # Add validation accuracy to the list
        accs.append(accuracy)
        
        # Print train results
        print(f"Validation of epoch {epoch+1} is completed in {time() - valid_tic:.3f} secs!\n")
        print(f"For epoch {epoch+1} the validation accuracy over the whole validation set is {accuracy:.2f}%\n")
        
        # Save the model with the best accuracy 
        if accuracy > best_accuracy:
            saveModel(ds_name, model)
            best_accuracy = accuracy
            
    # Save train times, validation times, and accuracy         
    torch.save(train_times, f'times2.0/{ds_name}_train_times')
    torch.save(valid_times, f'times2.0/{ds_name}_valid_times')
    torch.save(accs, f'times2.0/{ds_name}_accs')
