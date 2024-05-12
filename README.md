# README
This file is designed to guide the user on how to run both the test and training Python files. Additionally, you will find 'train.ipynb' and 'test.ipynb', which contain details on how to run the training and test files, respectively. Essentially, this README file is simply a compilation of the contents found in both of the aforementioned files.
## Training Case
We will begin by importing the required libraries.
```python
# Basic Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# Custom Imports
import transforms as T  # Custom transformations module
import utils  # Utility functions module
from dataloader import GbDataset, GbRawDataset, GbCropDataset  # Custom dataset loaders
from models import GbcNet  # Custom neural network model

# Other
import argparse
import os
import json
import copy
import neptune.new as neptune
from __future__ import print_function, division
from skimage import io, transform
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```
We'll also define a `device` variable to make our code flexible, enabling it to run on both GPU and CPU.
```python
# Determine device (CPU or GPU) for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
Since we'll be working within a notebook instead of executing the original train.py file from the command line, there's no need to inlcude the parser method. This also means we'll have to hardcode some value in the following segments as we won't be providing them through the command line.

**Defining the Image Transformation**

The following code snippet will define the image transformations to be applied to the images.

```python
# Initialize a list to store transformations
transforms = []

# Add resizing transformation to the list
transforms.append(T.Resize((224, 224)))

# Uncomment the line below to add random horizontal flip transformation with probability 0.25
# transforms.append(T.RandomHorizontalFlip(0.25))

# Add tensor conversion transformation to the list
transforms.append(T.ToTensor())

# Combine all transformations into a single composition
img_transforms = T.Compose(transforms)

# Define validation transformations using a composition
val_transforms = T.Compose([
    T.Resize((224, 224)),  # Resize images
    T.ToTensor()  # Convert images to tensors
])
```

**Loading the Dataset**

Next, we'll load in our dataset.

```python
# Open and load the metadata file
with open("data/roi_pred.json", "r") as f:
    df = json.load(f)

# Initialize lists to store training and validation labels
train_labels = []
val_labels = []

# Read and store training labels
train_file_path = os.path.join("data", "train.txt")
with open(train_file_path, "r") as f:
    for line in f.readlines():
        train_labels.append(line.strip())

# Read and store validation labels
val_file_path = os.path.join("data", "test.txt")
with open(val_file_path, "r") as f:
    for line in f.readlines():
        val_labels.append(line.strip())
```

**Visualizing the Dataset**


Once we have loaded our dataset, we can plot a few random samples of the training data. For example, we will plot the training images along with the ground truth ROI of the gallbladder.

According to the paper, the images are labeled as either 0, 1, or 2, corresponding to normal, benign, and malignant, respectively. Let's create a convenience function for this mapping.

```python
# Function to get the class label name based on class index
def get_class_label(class_index):
    if class_index == 0:
        return "Normal"
    elif class_index == 1:
        return "Benign"
    elif class_index == 2:
        return "Malignant"
    else:
        return "Unknown"
```

Now, let's read in and display a few of our images along with the ground truth ROI. Again, this is the area where the gallbladder resides, and where the model should classify as either normal, benign, or malignant.

```python
# Open and load the metadata file
with open("data/roi_pred.json", "r") as f:
    df = json.load(f)

# Function to plot images with bounding boxes
def plot_images_with_boxes(train_labels, df):
    plt.figure(figsize=(15, 9))
    for i in range(9):
        # Choose a random index
        random_index = random.randint(0, len(train_labels) - 1)
        img_name, img_class = train_labels[random_index][:11], train_labels[random_index][-1]

        # Load the image
        img = plt.imread('./data/imgs/' + img_name)

        # Plot the image
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.title(get_class_label(int(img_class)))
        plt.axis('off')

        # Get bounding box information from df dictionary
        bbox_info = df.get(img_name)
        if bbox_info:
            # Extract bounding box coordinates
            gold_box = bbox_info['Gold']
            x_min, y_min, x_max, y_max = gold_box[0], gold_box[1], gold_box[2], gold_box[3]

            # Create a rectangle patch
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=3, edgecolor='gold', facecolor='none')
            # Add the patch to the Axes
            plt.gca().add_patch(rect)

    plt.tight_layout()
    plt.show()

# Call the function to plot images with bounding boxes
plot_images_with_boxes(train_labels, df)
```

Lastly, we'll create our data loaders to separate the dataset for training purposes.

```python
# If ROI is specified, create a cropped dataset for validation
val_dataset = GbCropDataset('data/imgs', df, val_labels, to_blur=False, img_transforms=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=5)
```
**Initializing the Model**

Now, we'll initialize our model.

```python
net = GbcNet(num_cls=3, pretrain=False, att_mode='1') 
```

Following this, we can view our model architecture and examine its parameter count.

```python
# Get parameters that require gradients for optimization
params = [p for p in net.parameters() if p.requires_grad]

# Calculate total number of parameters in the network
total_params = sum(p.numel() for p in net.parameters())

# Print the total number of parameters
print("Total Parameters: ", total_params)
```

We can now set up the criterion and optimizer.

```python
# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer with stochastic gradient descent (SGD)
optimizer = optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=0.0005)

# Define the learning rate scheduler with step decay
lr_sched = StepLR(optimizer, step_size=5, gamma=0.8)
```
**Training the Model**

The code to actually train the model is shown below:

```python
from tqdm.notebook import tqdm

for epoch in tqdm(range(1), desc="Epochs"):
    if epoch < 10:
        train_dataset = GbDataset("data/imgs", df, train_labels, blur_kernel_size=(65,65), sigma=16, img_transforms=img_transforms)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=5)#, collate_fn=utils.collate_fn)
    elif epoch >= 10 and epoch < 15:
        train_dataset = GbDataset("data/imgs", df, train_labels, blur_kernel_size=(33,33), sigma=8, img_transforms=img_transforms)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
    elif epoch >= 15 and epoch < 20:
        train_dataset = GbDataset("data/imgs", df, train_labels, blur_kernel_size=(17,17), sigma=4, img_transforms=img_transforms)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
    elif epoch >= 20 and epoch < 25:
        train_dataset = GbDataset("data/imgs", df, train_labels, blur_kernel_size=(9,9), sigma=2, img_transforms=img_transforms)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
    elif epoch >= 25 and epoch < 30:
        train_dataset = GbDataset("data/imgs", df, train_labels, blur_kernel_size=(5,5), sigma=1, img_transforms=img_transforms)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
    else:
        train_dataset = GbDataset("data/imgs", df, train_labels, to_blur=False, img_transforms=img_transforms)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
    
    running_loss = 0.0
    total_step = len(train_loader)
    for images, targets, fnames in tqdm(train_loader, desc="Batches", leave=False):
        # images, targets = images.float().cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(images.float())
        loss = criterion(outputs.cpu(), targets.cpu())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss.append(running_loss/total_step)
   
    y_true, y_pred = [], []
    with torch.no_grad():
        net.eval()
        for images, targets, fname in tqdm(val_loader, desc="Validation", leave=False):
            # images, targets = images.float().cuda(), targets.cuda()
            if not False:
                images = images.squeeze(0)
                outputs = net(images.float())
                _, pred = torch.max(outputs, dim=1)
                pred_label = torch.max(pred)
                pred_idx = pred_label.item()
                pred_label = pred_label.unsqueeze(0)
                y_true.append(targets.tolist()[0][0])
                y_pred.append(pred_label.item())
            else:
                outputs = net(images.float())
                _, pred = torch.max(outputs, dim=1)
                pred_idx = pred.item()
                y_true.append(targets.tolist()[0])
                y_pred.append(pred.item())
        acc = accuracy_score(y_true, y_pred)
        cfm = confusion_matrix(y_true, y_pred)
        spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
        sens = cfm[2][2]/np.sum(cfm[2])
        print('Epoch: [{}/{}] Train-Loss: {:.4f} Val-Acc: {:.4f} Val-Spec: {:.4f} Val-Sens: {:.4f}'\
                .format(epoch+1, 1, train_loss[-1], acc, spec, sens))
```
**Evaluation**
Since we only trained the model for a single epoch, we cannot expect any meaningful inferences on the testing dataset.

Below, you can find the confusion matrix after running the model for a single epoch.

```python
# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for better visualization
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```

## Testing Case
Again, for the testing, we'll begin by importing the necessary libraries.

```python
# Basic Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# Custom Imports
import transforms as T  # Custom transformations module
import utils  # Utility functions module
from dataloader import GbDataset, GbRawDataset, GbCropDataset  # Custom dataset loaders
from models import GbcNet  # Custom neural network model

# Other
import argparse
import os
import json
import copy
import neptune.new as neptune
import pickle
from __future__ import print_function, division
from skimage import io, transform
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm.notebook import tqdm



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```
We'll also define a `device` variable in case we would like to run our model on the GPU.

```python
# Determine device (CPU or GPU) for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

**Image Transformation Pipeline & Loading in the Dataset**

Following this, we'll define the transformation pipeline and load in the test dataset.

```python
# List to store transformation functions
transforms = []

# Resize the image to specified width and height
transforms.append(T.Resize((224, 224)))

# Optional: Uncomment the following line to apply random horizontal flip with a probability of 25%
# transforms.append(T.RandomHorizontalFlip(0.25))

# Convert the image to a PyTorch tensor
transforms.append(T.ToTensor())

# Combine all transformations into a single transformation pipeline
img_transforms = T.Compose(transforms)

# Define transformation for validation data
val_transforms = T.Compose([
    # Resize the image to specified width and height
    T.Resize((224, 224)),
    # Convert the image to a PyTorch tensor
    T.ToTensor()
])
```

```python
# Load metadata from the specified JSON file
with open("data/roi_pred.json", "r") as f:
    df = json.load(f)

# Load validation labels from the test set file
val_labels = []
v_fname = os.path.join("data", "test.txt")
with open(v_fname, "r") as f:
    for line in f.readlines():
        val_labels.append(line.strip())


# Create a dataset with ROI cropping
val_dataset = GbCropDataset("data/imgs", df, val_labels, to_blur=True, sigma=0, p=0.15, img_transforms=val_transforms)

# Create a DataLoader for the validation dataset
# Set batch_size=1 since we're using one image at a time for validation
# Set shuffle=True to shuffle the data before each epoch
# Set num_workers to the number of subprocesses to use for data loading
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
```

**Model Initialization**

For this part, we'll simply load in the pre-trained model.

```python
# Create an instance of the GbcNet model
# num_cls: Number of classes in your classification task
# pretrain: Whether to use pre-trained weights (assuming True/False)
# att_mode: Attention mode used in the model
# head: Head type used in the model
net = GbcNet(num_cls=3, pretrain=False, att_mode='1', head='2')

# Load the model's state dictionary from the specified path
# net.load_state_dict(torch.load('gbcnet.pth'))
net.load_state_dict(torch.load('gbcnet.pth', map_location=device))

# Move the model to the GPU and set it to operate with float data type
# net.net = net.net.float().cuda()
```
**Model Evaluation on Validation Dataset**

```python
# Set the model to evaluation mode
net.eval()

# Lists to store true labels, predicted labels, and prediction scores
y_true = []
y_pred = []
score_dmp = []

# Wrap the loop with tqdm for progress monitoring
for images, targets, filenames in tqdm(val_loader, desc="Validation", unit="batch"):
    # Move images and targets to the GPU and convert them to float type
    # images, targets = images.float().cuda(), targets.cuda()
    
    # Get the filename of the current image
    cam_img_name = filenames[0]
    
    # Check if ROI cropping is enabled
    if not False:
        # Squeeze the batch dimension if present
        images = images.squeeze(0)
        
        # Forward pass through the network
        outputs = net(images.float())
        
        # Get the predicted class label
        _, pred = torch.max(outputs, dim=1)
        pred_label = torch.max(pred)
        pred_idx = pred_label.item()
        pred_label = pred_label.unsqueeze(0)
        idx = torch.argmax(pred)
        
        # Append true and predicted labels to the lists
        y_true.append(targets.tolist()[0][0])
        y_pred.append(pred_label.item())
        
        # Append true label and prediction score to score_dmp list
        score_dmp.append([y_true[-1], outputs[idx.item()].tolist()])
    else:
        # Forward pass through the network
        outputs = net(images.float())
        
        # Get the predicted class label
        _, pred = torch.max(outputs, dim=1)
        pred_idx = pred.item()
        
        # Append true and predicted labels to the lists
        y_true.append(targets.tolist()[0])
        y_pred.append(pred.item())
        
        # Append true label and prediction score to score_dmp list
        score_dmp.append([y_true[-1], outputs[0].tolist()])
```

**Confusion Matrix**

The below snippet of code is used to generate the confusion matrix.

```python
# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for better visualization
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```

**Inference**

Finally, we can plot the model's prediction against the ground truth.

```python
# Function to get the class label name based on class index
def get_class_label(class_index):
    if class_index == 0:
        return "Normal"
    elif class_index == 1:
        return "Benign"
    elif class_index == 2:
        return "Malignant"
    else:
        return "Unknown"
```
```python
def plot_images_with_boxes(train_labels, df):
    plt.figure(figsize=(15, 9))
    for i in range(9):
        # Choose a random index
        random_index = random.randint(0, len(train_labels) - 1)
        img_name, img_class = train_labels[random_index][:11], train_labels[random_index][-1]

        # Load the image
        img = plt.imread('./data/imgs/' + img_name)

        # Plot the image
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.title(get_class_label(int(img_class)))
        plt.axis('off')

        # Get bounding box information from df dictionary
        bbox_info = df.get(img_name)
        if bbox_info:
            # Extract ground truth bounding box coordinates
            gold_box = bbox_info['Gold']
            x_min_gt, y_min_gt, x_max_gt, y_max_gt = gold_box[0], gold_box[1], gold_box[2], gold_box[3]

            # Create a rectangle patch for ground truth
            rect_gt = patches.Rectangle((x_min_gt, y_min_gt), x_max_gt - x_min_gt, y_max_gt - y_min_gt, linewidth=3, edgecolor='gold', facecolor='none')
            # Add the ground truth patch to the Axes
            plt.gca().add_patch(rect_gt)

            # Extract predicted bounding box coordinates
            pred_boxes = bbox_info['Boxes']
            for pred_box in pred_boxes:
                x_min_pred, y_min_pred, x_max_pred, y_max_pred = pred_box[0], pred_box[1], pred_box[2], pred_box[3]

                # Create a rectangle patch for predicted bounding box
                rect_pred = patches.Rectangle((x_min_pred, y_min_pred), x_max_pred - x_min_pred, y_max_pred - y_min_pred, linewidth=3, edgecolor='red', facecolor='none')
                # Add the predicted patch to the Axes
                plt.gca().add_patch(rect_pred)

    plt.tight_layout()
    plt.show()

# Call the function to plot images with both ground truth and predicted bounding boxes
plot_images_with_boxes(val_labels, df)
```
