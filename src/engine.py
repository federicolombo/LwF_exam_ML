import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader

import os 
import csv 
from tqdm import tqdm 
from datetime import datetime
from typing import Dict, Callable

from sklearn.metrics import average_precision_score
import numpy as np 

class TrainingEgine:
    
    def __init__(self, new_model: nn.Module, 
                 old_model: nn.Module, 
                 train_dataloader: DataLoader, 
                 test_dataloader: DataLoader, 
                 optimizer: torch.optim.Optimizer, 
                 loss_old_task: Callable[[torch.Tensor, torch.Tensor, int], float], 
                 loss_new_task: nn.Module, 
                 epochs: int, 
                 device: torch.device, 
                 taskName: str,  
                 lamda: int) -> Dict[str, list]:
        self.new_model = new_model
        self.old_model = old_model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_old_task = loss_old_task
        self.loss_new_task = loss_new_task
        self.epochs = epochs
        self.device = device
        self.taskName = taskName 
        self.lamda = lamda
        
        self.new_model = self.new_model.to(self.device)
        
        self.output_file = self._generate_ouptut_file_name()
        
    def _generate_ouptut_file_name(self) -> str:
        """
        Generate a file name based on the task name and the current timestamp, saving it in the results folder.
        """
        
        timestamp = datetime.now().strftime("d%d_m%m_h%H_m%M")
        filename = f"{self.taskName}_training_results_{timestamp}.csv"
        folder_path = "../results"
        
        os.makedirs(folder_path, exist_ok=True)
        
        return os.path.join(folder_path, filename)
        
    def _old_task_classes(self, model: nn.Module) -> int:
        out_features = model.classifier.out_features
        return out_features
    
    def _train_step(self):
        
        self.new_model.train()
        
        train_loss = 0.0 
        train_loss_old_task = 0.0 
        train_acc = 0.0
        
        y_tr = []
        y_preds = []
        
        num_old_tasks = self._old_task_classes(self.old_model)
        
        for X, y in self.train_dataloader:
            
            X, y = X.to(self.device), y.to(self.device)
            
            y_pred_new = self.new_model(X)
            logits_old_task = y_pred_new[:, :num_old_tasks]
            logits_new_task = y_pred_new[:, num_old_tasks:]
            
            y_pred_old = self.old_model(X)
            
            loss_nt = self.loss_new_task(logits_new_task, y)
            loss_ot = self.loss_old_task(logits_old_task, y_pred_old)
            
            
            loss = loss_nt + loss_ot * self.lamda # lamda is an hyperparameter that balances the losses of the old task, in the original paper is set to 1   
            train_loss += loss.item()
            train_loss_old_task += loss_ot.item()
            
            self.optimizer.zero_grad()
            
            loss.backward()
            
            self.optimizer.step()
            
            if self.taskName == "voc":
                y_pred_prob = torch.sigmoid(logits_new_task).detach().cpu().numpy()
                y_true = y.detach().cpu().numpy()
                
                y_tr.append(y_true)
                y_preds.append(y_pred_prob)
                
            if self.taskName == "cub":
                y_pred_class = torch.argmax(torch.softmax(logits_new_task, dim=1), dim=1)
                train_acc += (y == y_pred_class).sum().item() / len(y)
    
    
        if self.taskName == "voc":
            y_tr = np.vstack(y_tr)
            y_preds = np.vstack(y_preds)
            
            train_map = average_precision_score(y_tr, y_preds, average="samples")
            
            train_loss /= len(self.train_dataloader)
            train_loss_old_task /= len(self.train_dataloader)
            
            return train_loss, train_map, train_loss_old_task
        
        if self.taskName == 'cub':
            train_loss /= len(self.train_dataloader)
            train_loss_old_task /= len(self.train_dataloader)
            train_acc /= len(self.train_dataloader)
            
            return train_loss, train_acc, train_loss_old_task
            

    def _test_step(self):
        
        self.new_model.eval()
        
        test_loss = 0.0 
        test_acc = 0.0 
        
        y_tr = []
        y_preds = []
        
        num_old_tasks = self._old_task_classes(self.old_model)
        
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                
                X, y = X.to(self.device), y.to(self.device)
                
                y_pred = self.new_model(X)
                logits_new_task = y_pred[:, num_old_tasks:]
                
                loss = self.loss_new_task(logits_new_task, y)
                test_loss += loss.item()
                
                if self.taskName == "voc":
                    y_pred_prob = torch.sigmoid(logits_new_task).detach().cpu().numpy()
                    y_true = y.detach().cpu().numpy()
                    
                    y_tr.append(y_true)
                    y_preds.append(y_pred_prob)
                
                if self.taskName == "cub":
                    y_pred_class = torch.argmax(torch.softmax(logits_new_task, dim=1), dim=1)
                    test_acc += (y == y_pred_class).sum().item() / len(y)
                    
        if self.taskName == "voc":
            y_tr = np.vstack(y_tr)
            y_preds = np.vstack(y_preds)
            test_map = average_precision_score(y_tr, y_preds, average="samples")
            test_loss /= len(self.test_dataloader)
            
            return test_loss, test_map
        
        if self.taskName == "cub":
            test_loss /= len(self.test_dataloader)
            test_acc /= len(self.test_dataloader)
            
            return test_loss, test_acc
                    

    def training(self) -> Dict:
        """        
        Train the model and return the training results.
        
        Returns:
            Dict: A dictionary containing the following keys:
                - "train_loss": A list of training losses.
                - "train_metric": A list of training metrics.
                - "test_loss": A list of test losses.
                - "test_metric": A list of test metrics.
                - "train_loss_old_task": A list of training losses for the old task. 
        """
        
        results = {
            "train_loss": [],
            "train_metric": [],
            "test_loss": [],
            "test_metric": [],
            "train_loss_old_task": []
        }
        
        if self.taskName == "voc":
            tr_metric = "Train mAP"
            ts_metric = "Test mAP"
        if self.taskName == "cub":
            tr_metric = "Train Accuracy"
            ts_metric = "Test Accuracy"
            
        with open(self.output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train Loss Old Task', tr_metric, 'Test Loss', ts_metric])
            
            for epoch in tqdm(range(self.epochs), desc="Training Progress"):
                train_loss, train_metric, train_loss_old_task = self._train_step()
                test_loss, test_metric = self._test_step()
                
                print(
                    f"\n--------- Epoch: {epoch + 1} ----------\n",
                    f"Training results:\n",
                    f"\tTrain Loss: {train_loss:.2f}\n",
                    f"\tTrain Loss Old Task: {train_loss_old_task:.2f}\n",
                    f"\t{tr_metric}: {train_metric * 100:.2f}%\n\n"
                    f"Validation Results:\n"
                    f"\tTest Loss: {test_loss:.2f}\n",
                    f"\t{ts_metric}: {test_metric * 100:.2f}%\n",
                "-------------------------------------------\n"
                    
                )
                
                results["train_loss"].append(train_loss)
                results["train_metric"].append(train_metric)
                results["test_loss"].append(test_loss)
                results["test_metric"].append(test_metric)  
                results["train_loss_old_task"].append(train_loss_old_task)
                
                writer.writerow([epoch + 1, train_loss, train_loss_old_task, train_metric * 100, test_loss, test_metric * 100])
                
        
        return results
