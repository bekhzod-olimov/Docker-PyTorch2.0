import timm, torch
from tqdm import tqdm
from time import time

def saveModel(ds_name, model):
    
    path = f"./{ds_name}_best_model.pth"
    torch.save(model.state_dict(), path)

def validation(model, val_dl, device, ds_name):
    
    model.eval()
    accuracy, total = 0, 0

    with torch.no_grad():
        
        for i, batch in tqdm(enumerate(val_dl)):

            images, labels = batch
            
            # get 3 channels for mnist dataset
            if ds_name == "mnist":
                images_copy = images
                images = torch.cat((images, images_copy), dim=1)
                images = torch.cat((images, images_copy), dim=1)
                
            images, labels = images.to(device), labels.to(device)
            
            images, labels = images.to(device), labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    
    return accuracy
    
def train(model, tr_dl, val_dl, num_classes, criterion, optimizer, device, epochs, best_accuracy, ds_name):

    # Define your execution device
    print(f"The model will be running on {device} device")
    model.to(device)
    train_times, valid_times = [], []
    
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
        print(f"Validation of epoch {epoch+1} is completed in {time() - valid_tic:.3f} secs!\n")
        
        print(f"For epoch {epoch+1} the validation accuracy over the whole validation set is {accuracy:.2f}%\n")
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(ds_name, model)
            best_accuracy = accuracy
            
    torch.save(train_times, f'{ds_name}_train_times')
    torch.save(valid_times, f'{ds_name}_valid_times')
            
        