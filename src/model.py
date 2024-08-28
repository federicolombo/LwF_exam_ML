import torch 
import torch.nn as nn 
from torchvision import models
from collections import OrderedDict
from pathlib import Path 

class LwFNet(nn.Module):
    def __init__(self, pretrained_model_path: Path, old_num_tasks: int=365):
        super(LwFNet, self).__init__()
        
        # Load alexnet model from torchvision
        self.model = models.alexnet()
        
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, old_num_tasks)
        self.classifier = self.model.classifier[6]
        
        if pretrained_model_path:
            checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('features.module.', 'features.')
                new_state_dict[name] = v
                    
            self.model.load_state_dict(new_state_dict, strict=False)

              
    def forward(self, x):
        return self.model(x)
    
    def update_tasks(self, new_tasks: int):
        """
        Update the last linear layer of the model to include new tasks.
        
        Args:
            new_tasks (int): The number of new tasks to add.
        """
        
        old_in_features = self.classifier.in_features
        old_out_features = self.classifier.out_features
        weights = self.classifier.weight.data
        
        self.model.classifier[6] = nn.Linear(old_in_features, old_out_features + new_tasks)
        
        self.classifier = self.model.classifier[6]
        self.classifier.weight.data[:old_out_features] = weights
        
        print(f"""\nUpdated the last linear layer of the model. 
              \n\tOld out_features: {old_out_features} 
              \n\tNew out_features: {old_out_features + new_tasks}\n\n""")
        