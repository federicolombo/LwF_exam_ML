import torch 
from torchvision import transforms
import torch.optim as optim 

import copy
from pathlib import Path 
from datetime import datetime
import urllib
import argparse
import os 

from engine import TrainingEgine
from utils import distillation_loss, save_model, voc_plot_training_results, cub_plot_training_results
from get_cub_data import cub_dataloader
from get_voc_data import voc_dataloader
from model import LwFNet

# Define the argument parser
parser = argparse.ArgumentParser(description="Training script for Learning Without Forgetting model.")

parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading.")
parser.add_argument("--save_model_results", type=bool, default=True, help="Flag to save model results.")
parser.add_argument("--task", type=str, default="cub", choices=["voc", "cub"], help="Model to train.")
parser.add_argument("--download_data", default=False, action='store_true', help="Flag to download data if not already available.")

args = parser.parse_args()

# Assign parsed arguments to variables
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_WORKERS = args.num_workers
SAVE_MODEL_RESULTS = args.save_model_results
TASK = args.task
DOWNLOAD_DATA = args.download_data

# Set the path to the datasets and the directory to save the models
path_to_voc = Path("/Users/coli/dataset/Dataset_ML_Project/LwF")
path_to_cub = Path("/Users/coli/dataset")
path_to_save_model = Path("../models")
model_path = Path("../models/alexnet_places365.pth.tar")

if not os.path.exists(path_to_save_model):
  os.makedirs(path_to_save_model)
  print(f"Directory '{path_to_save_model}' created.")
  
if not os.path.exists("data"):
  os.makedirs("data")
  print(f"Directory data created.")

if not model_path.exists():
  try:
    urllib.request.urlretrieve("http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar", model_path)
  except Exception as e:
    print(f"Error downloading the model: {e}")

# Set the device 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"Current device: {device}")

# Implement some custom transformations for the dataset 
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224, 224), padding=5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if DOWNLOAD_DATA:
    print(f"Downloading data for task: {TASK}")
    
    if TASK == "voc":
        train_dataloader = voc_dataloader(data_path=path_to_voc,
                                            custom_transform=train_transform,
                                            batch_size=BATCH_SIZE,
                                            image_set="train",
                                            num_workers=NUM_WORKERS,
                                            label_group=None,
                                            download=True
                                              )
        
    elif TASK == "cub":
        train_dataloader = cub_dataloader(data_path=path_to_cub,
                                        train=True,
                                        img_transform=train_transform,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        download=True
                                        
                                        )
            
    print("Data download complete.")
    
# Define the model 
model_path = Path("../models/alexnet_places365.pth.tar")
model = LwFNet(pretrained_model_path=model_path)

# Make a copy of the model to retain the old task
old_model = copy.deepcopy(model)
old_model = old_model.to(device)

# Define optimizer for the new task
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

if TASK == "voc":
    
    loss_new_task = torch.nn.BCEWithLogitsLoss()
    labels_new_tasks = {
        "transport_voc1" : ['train', 'bus', 'motorbike', 'aeroplane', 'bicycle', 'boat', 'car'],
        "animals_voc2" : ['bird', 'horse', 'cow', 'dog', 'cat', 'sheep', 'person'],
        "objects_voc3" : ['bottle', 'pottedplant', 'tvmonitor', 'chair', 'sofa', 'diningtable']
        }
    final_results = {}
    
    for task in labels_new_tasks:
        print(f"\nTraining on task: {task}\n")
        
        list_of_labels = labels_new_tasks[task]
        
        # Update dataloaders for the current task
        train_dataloader = voc_dataloader(data_path=path_to_voc,
                                             custom_transform=train_transform,
                                            batch_size=BATCH_SIZE,
                                            image_set="train",
                                            num_workers=NUM_WORKERS,
                                            label_group=list_of_labels
                                              )
        test_dataloader = voc_dataloader(data_path=path_to_voc,
                                            custom_transform=test_transform,
                                            batch_size=BATCH_SIZE,
                                            image_set="val",
                                            num_workers=NUM_WORKERS,
                                            label_group=list_of_labels
                                              )
        
        num_new_task = len(list_of_labels)
        model.update_tasks(num_new_task)
        
        engine = TrainingEgine(
            model, 
            old_model, 
            train_dataloader, 
            test_dataloader, 
            optimizer, 
            distillation_loss,
            loss_new_task,
            NUM_EPOCHS, 
            device, 
            TASK,
            1)
        
        task_results = engine.training()
        
        # Save the results and models after each task 
        final_results[task] = task_results
        old_model = copy.deepcopy(model)
        
    if SAVE_MODEL_RESULTS:
        now = datetime.now()
        day = now.strftime("%d")
        month = now.strftime("%m")
        hour = now.strftime("%H")
        minute = now.strftime("%M")
        
        model_name = f"LwF_{TASK}_{day}D_{month}M_{hour}h_{minute}m.pth"
        
        save_model(model=model,
                target_dir=path_to_save_model,
                model_name=model_name
                )

        
    voc_plot_training_results(final_results, list(labels_new_tasks.keys()))
    
    
if TASK == "cub":
    path_to_cub = Path("/Users/coli/dataset")
    loss_new_task = torch.nn.CrossEntropyLoss()
    train_dataloader = cub_dataloader(data_path=path_to_cub,
                                      train=True,
                                      img_transform=train_transform,
                                      batch_size=BATCH_SIZE,
                                      num_workers=NUM_WORKERS
                                      )
    test_dataloader = cub_dataloader(data_path=path_to_cub,
                                        train=False,
                                        img_transform=test_transform,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS
                                        )
    
    model.update_tasks(200)
    engine = TrainingEgine(
        model, 
        old_model, 
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        distillation_loss,
        loss_new_task,
        NUM_EPOCHS, 
        device, 
        TASK,
        1)
    
    task_results = engine.training()
    
    if SAVE_MODEL_RESULTS:
        now = datetime.now()
        day = now.strftime("%d")
        month = now.strftime("%m")
        hour = now.strftime("%H")
        minute = now.strftime("%M")
        
        model_name = f"LwF_{TASK}_{day}D_{month}M_{hour}h_{minute}m.pth"
        
        save_model(model=model,
                target_dir=path_to_save_model,
                model_name=model_name
                )
    
    cub_plot_training_results(task_results)
    
    