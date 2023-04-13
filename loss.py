import torch.nn as nn
import torch
class Unified_Adversarial_Loss(nn.Module):
    def __init__(self, bce_weight=0.5, mse_weight=0.5, hard_negative_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.hard_negative_weight = hard_negative_weight
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):

        # Calculate the binary cross-entropy loss for the classification part
        bce_loss = self.bce_loss(output[:, :2].clone(), target[:, :2].clone())

        # Calculate the mean squared error loss for the regression part
        mse_loss = self.mse_loss(output[:, 2:].clone(), target[:, 2:].clone())

        # Calculate the hard negative loss for the classification part
        target_cls_int = torch.argmax(target[:, :2].clone(), dim=1)  # Convert the one-hot encoding to integers
        output_cls_int = torch.argmax(output[:, :2].clone(), dim=1)  # Convert the one-hot encoding to integers
        hard_negative_mask = (target_cls_int != output_cls_int) # Create a mask for hard negative examples
        hard_negative_mask = hard_negative_mask.float()  # Convert the mask to a float tensor
        hard_negative_loss = torch.mean(hard_negative_mask) * self.hard_negative_weight * bce_loss

        # Combine the three losses
        
        loss = self.bce_weight * bce_loss + self.mse_weight * mse_loss + hard_negative_loss

        return loss
