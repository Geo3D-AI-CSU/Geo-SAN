import torch

class GradNorm_2loss:
    def __init__(self, alpha=1.0, gamma=1.0, device='cuda'):
        """
        Initialise the GradNorm calculator.
        """
        self.alpha = alpha  # Level loss initial weighting
        self.gamma = gamma  # Gradient loss initial weighting
        self.device = device
        # Weights are stored in a tensor.
        self.loss_weights = torch.tensor([alpha, gamma], dtype=torch.float32, device=device)

    def compute_loss(self, level_loss, gradient_loss):
        """
        Calculate the weighted total loss based on the updated loss weights.
        """
        return self.loss_weights[0] * level_loss + self.loss_weights[1] * gradient_loss

    def update_weights(self, level_loss, gradient_loss, model):
        """
        Dynamically update the weights of the loss function according to the GradNorm algorithm.
        """
        # Compute the gradient norm for each loss
        level_grad_norm = self.compute_grad_norm(level_loss, model)
        grad_grad_norm = self.compute_grad_norm(gradient_loss, model)

        grad_norms = torch.tensor([level_grad_norm, grad_grad_norm], device=self.device)

        # Normalised gradient norm
        normed_grad_norms = grad_norms / grad_norms.mean()
        grad_ratio = normed_grad_norms / (grad_norms + 1e-8)  # Preventing zero removal

        # Update weighting
        self.loss_weights[:len(grad_ratio)] = self.loss_weights[:len(grad_ratio)] * grad_ratio
        self.loss_weights = self.loss_weights / self.loss_weights.sum()  

        return self.loss_weights

    def compute_grad_norm(self, loss, model):
        """
        Compute the gradient norm for each loss.
        """
        loss.backward(retain_graph=True)  # Maintain computational graph
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        return grad_norm ** 0.5

class GradNorm_3loss:
    def __init__(self, alpha=1.0, gamma=1.0, delta=1.0, device='cuda'):
        self.alpha = alpha 
        self.gamma = gamma  
        self.delta = delta  
        self.device = device
        # The weights are stored in a tensor, initialised as [alpha, gamma, delta]
        self.loss_weights = torch.tensor([alpha, gamma, delta], dtype=torch.float32, device=device)

    def compute_loss(self, level_loss, gradient_loss, scalar_loss):

        return self.loss_weights[0] * level_loss + self.loss_weights[1] * gradient_loss + self.loss_weights[2] * scalar_loss

    def update_weights(self, level_loss, gradient_loss, scalar_loss, model):

        level_grad_norm = self.compute_grad_norm(level_loss, model)
        grad_grad_norm = self.compute_grad_norm(gradient_loss, model)
        scalar_grad_norm = self.compute_grad_norm(scalar_loss, model)
        grad_norms = torch.tensor([level_grad_norm, grad_grad_norm, scalar_grad_norm], device=self.device)

        normed_grad_norms = grad_norms / grad_norms.mean()
        grad_ratio = normed_grad_norms / (grad_norms + 1e-8) 


        self.loss_weights[:len(grad_ratio)] = self.loss_weights[:len(grad_ratio)] * grad_ratio
        self.loss_weights = self.loss_weights / self.loss_weights.sum() 

        return self.loss_weights

    def compute_grad_norm(self, loss, model):
        loss.backward(retain_graph=True)  
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        return grad_norm ** 0.5
