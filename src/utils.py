import torch 
from pathlib import Path
import numpy as np 
import torch.nn.functional as F
import matplotlib.pyplot as plt

def distillation_loss(old_logits: torch.Tensor, new_logits: torch.Tensor, temperature=2) -> float:
    """
    Compute the distillation loss as described in the paper.
    
    Parameters:
    - old_logits (Tensor): The output probabilities from the student network.
    - new_logits (Tensor): The ground truth labels (not used in distillation loss directly).
    - temperature (float): The temperature parameter for smoothing.
    
    Returns:
    - loss (Tensor): The computed distillation loss.
    """
    # Softening the probabilities
    soft_target = F.softmax(old_logits / temperature, dim=1)
    soft_output = F.log_softmax(new_logits / temperature, dim=1)
    
    # Compute the cross-entropy loss between the softened teacher output and the student output
    distillation_loss = F.kl_div(soft_output, soft_target, reduction='batchmean') * (temperature ** 2)
    
    return distillation_loss

def save_model(model: torch.nn,
               target_dir: str,
               model_name: str):
    """ 
    Saves a PyTorch model to a target directory.
    
    Args:
        model: (torch.nn) The model to save.
        target_dir: (str) The target directory.
        model_name: (str) The name of the model. Should include 
            either .pt or .pth extension.
    """
    
    # Create the target directory 
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create the model path 
    assert model_name.endswith(".pt") or model_name.endswith(".pth"), "Model name should include .pt or .pth extension."
    model_save_path = target_dir_path / model_name 
    
    # Save the model 
    print(f"Saving the model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)


def voc_plot_training_results(final_results, task_names):
    """
    Plot the training results for the VOC dataset.

    Args:
        final_results: (dict) The final results from the training.
        task_names (list): The names of the tasks.
    """
    train_metrics = []
    test_metrics = []
    old_task_losses = []
    
    # Extracting the final metrics and old task losses for each task
    for task in task_names:
        train_metrics.append(final_results[task]['train_metric'][-1])
        test_metrics.append(final_results[task]['test_metric'][-1])
        old_task_losses.append(final_results[task]['train_loss_old_task'][-1])

    x = range(len(task_names))
    
    # Creating the plot
    plt.figure(figsize=(14, 6))
    
    # Plotting Train and Test Metrics
    plt.subplot(1, 2, 1)
    plt.plot(x, train_metrics, marker='o', label='Train Metric')
    plt.plot(x, test_metrics, marker='o', label='Test Metric')
    plt.xticks(x, task_names)
    plt.xlabel('Tasks (VOC Splits)')
    plt.ylabel('Metric')
    plt.title('Train vs Test Metric Across Tasks')
    plt.legend()
    
    # Plotting Old Task Loss
    plt.subplot(1, 2, 2)
    plt.plot(x, old_task_losses, marker='o', color='r', label='Old Task Loss')
    plt.xticks(x, task_names)
    plt.xlabel('Tasks (VOC Splits)')
    plt.ylabel('Loss')
    plt.title('Old Task Loss Across Tasks')
    plt.legend()
    
    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    
def cub_plot_training_results(final_results):
    """
    Plot the training results for the CUB dataset.

    Args:
        final_results (dict): The final results from the training.
    """
    
    train_accuracies = final_results['train_metric']
    test_accuracies = final_results['test_metric']
    old_task_losses = final_results['train_loss_old_task']
    epochs = range(1, len(train_accuracies) + 1)
    
    # Creating the plot
    plt.figure(figsize=(14, 6))
    
    # Plotting Train and Test Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(epochs, test_accuracies, marker='o', label='Test Accuracy')
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy Across Epochs')
    plt.legend()
    
    # Plotting Old Task Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, old_task_losses, marker='o', color='r', label='Old Task Loss')
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Old Task Loss Across Epochs')
    plt.legend()
    
    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
