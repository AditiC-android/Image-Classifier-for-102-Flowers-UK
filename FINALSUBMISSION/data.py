import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

def augment(path = "./flowers"):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    ##loading the literal data
    # TODO: Define transforms for the training data and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    
    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle = False)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=20, shuffle = False)
    
    return trainloader, testloader, validloader



def preparing(option = 'densenet121', dropout = 0.2, hidden_units = 256, lr = 0.003, source = 'gpu'):
    if torch.cuda.isavailable() and source = 'gpu':
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
            
     
    if option == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif option == 'vgg16':
        model = models.densenet169(pretrained = True)
    else:
        print("This is not valid. Try densenet121 or vgg16")
              
    for param in model.parameters():
    param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)
              
    return model, criterion, optimizer

def train(model, criterion, optimizer, epochs = 2, print_every=5, trainloader, validloader, source = 'gpu'):
    if torch.cuda.isavailable() and source = 'gpu':
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    early_stopping_patience = 4  # Stop training if no improvement for 4 validations

    # Initialize early stopping variables
    best_valid_loss = float('inf')  # Set to infinity so the first validation loss will always be better
    steps_no_improve = 0  # Counter to track how many validations without improvement

    for e in range(epochs):
        running_loss = 0
        print(f"Starting Epoch {e+1}/{epochs}...")

        ## Running loss tracks the loss
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            ## Forward pass
            logps = model(images)
            loss = criterion(logps, labels)

            ## Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if steps % print_every == 0:
                # Perform validation
                valid_loss = 0
                accuracy = 0

                ## Turn off gradients for validation
                with torch.no_grad():
                    model.eval()  ## Set model to evaluation mode
                    for val_images, val_labels in validloader:
                        val_images, val_labels = val_images.to(device), val_labels.to(device)
                        val_logps = model(val_images)
                        valid_loss += criterion(val_logps, val_labels).item()

                        ps = torch.exp(val_logps)
                        val_top_p, val_top_class = ps.topk(1, dim=1)
                        equals = val_top_class == val_labels.view(*val_top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                avg_valid_loss = valid_loss / len(validloader)
                avg_accuracy = accuracy / len(validloader)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Step: {}".format(steps),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Valid Loss: {:.3f}.. ".format(avg_valid_loss),
                      "Valid Accuracy: {:.3f}".format(avg_accuracy))

                model.train()  
                running_loss = 0  

                # Early Stopping Check
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    steps_no_improve = 0  
                    best_epoch = e + 1  
                    best_model_state = model.state_dict()  # Save best model state
                else:
                    steps_no_improve += 1  # Increment counter if no improvement

                # Trigger early stopping
                if steps_no_improve >= early_stopping_patience:
                    print("Early stopping triggered after {} validations!".format(steps_no_improve))
                    break  # Exit the training loop

        # Stop outer loop if early stopping is triggered
        if steps_no_improve >= early_stopping_patience:
            break

    print("Training complete")
    print("Total Epochs: {}".format(epochs))
    print("Number of Steps: {}".format(steps))

def save_checkpoint(check = 'checkpoint.pth', option = 'densenet121', hidden_units = 256, dropout = 0.2, lr = 0.003):
    model.class_to_idx = train_data.class_to_idx
    torch.save({'structure' :'densenet121',
                'model_state_dict':best_model_state,
                'dropout':0.2,
                'inputs':model.classifier[0].in_features,
                'hidden_units':256,
                'output': 102,
                'class_to_idx':model.class_to_idx},
                'checkpoint.pth')
    print("Checkpoint saved.")
    
def load_model(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))  
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(checkpoint['inputs'], checkpoint['hidden_units']),  
        nn.ReLU(),
        nn.Dropout(checkpoint['dropout']),
        nn.Linear(checkpoint['hidden_units'], checkpoint['output']),  
        nn.LogSoftmax(dim=1)
)


    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
    

def process_image(image_path):
    img_pil = Image.open(image_path).convert("RGB")  # Ensure image is RGB
    
    # Define the transformation pipeline
    transform_pipeline = transforms.Compose([
        transforms.Resize(256),              # Resize shortest side to 256
        transforms.CenterCrop(224),         # Center crop to 224x224
        transforms.ToTensor(),              # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                             std=[0.229, 0.224, 0.225])   # and standard deviation
    ])
    
    # Apply the transformations
    img_tensor = transform_pipeline(img_pil)
    
    return img_tensor

    
    
def predict(image_path, model, topk=5, img_tensor, source = 'gpu'):
    ''' Predict the top K classes of an image using a trained model. '''
    if torch.cuda.isavailable() and source = 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        model.eval()
        output = model(img_tensor)
        # Apply softmax to get probabilities
        ps = torch.softmax(output, dim=1)
        
        # Get the top K classes and probabilities
        probs, indices = ps.topk(topk, dim=1)
        
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class[idx.item()] for idx in indices[0]]
        
        return probs[0].cpu().numpy(), classes
        
    
    
    




              
              

          
    
    
