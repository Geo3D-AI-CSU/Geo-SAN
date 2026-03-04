import torch
import torch.nn as nn
import numpy as np

# Stratigraphic column information
STRATIGRAPHIC_COLUMN = {
    "units": [
        {"name": "T2B2", "rock_unit": 1, "top_interface": 13, "bottom_interface": 12},
        {"name": "T2B1", "rock_unit": 2, "top_interface": 12, "bottom_interface": 11},
        {"name": "T1b", "rock_unit": 3, "top_interface": 11, "bottom_interface": 10},
        {"name": "T1m", "rock_unit": 4, "top_interface": 10, "bottom_interface": 9},
        {"name": "P1m", "rock_unit": 5, "top_interface": 9, "bottom_interface": 8},
        {"name": "P1q", "rock_unit": 6, "top_interface": 8, "bottom_interface": 7},
        {"name": "C3", "rock_unit": 7, "top_interface": 7, "bottom_interface": 6},
        {"name": "C2", "rock_unit": 8, "top_interface": 6, "bottom_interface": 5},
        {"name": "C1", "rock_unit": 9, "top_interface": 5, "bottom_interface": 4},
        {"name": "D3", "rock_unit": 10, "top_interface": 4, "bottom_interface": 3},
        {"name": "D2d", "rock_unit": 11, "top_interface": 3, "bottom_interface": 2},
        {"name": "D1y", "rock_unit": 12, "top_interface": 2, "bottom_interface": 1},
        {"name": "D1n", "rock_unit": 13, "top_interface": 1, "bottom_interface": 0}
    ],
    "interfaces": [
        {"name": "Q-T2B2", "type": "conformable", "age": 13, "level": -1898},
        {"name": "T2B2-T2B1", "type": "conformable", "age": 12, "level": -3734},
        {"name": "T2B1-T1b", "type": "conformable", "age": 11, "level": -4872},
        {"name": "T1b-T1m", "type": "conformable", "age": 10, "level": -6772},
        {"name": "T1m-P1m", "type": "unconformity", "age": 9, "level": -7233},
        {"name": "P1m-P1q", "type": "conformable", "age": 8, "level": -8214},
        {"name": "P1q-C3", "type": "unconformity", "age": 7, "level": -8429},
        {"name": "C3-C2", "type": "conformable", "age": 6, "level": -9145},
        {"name": "C2-C1", "type": "conformable", "age": 5, "level": -9674},
        {"name": "C1-D3", "type": "unconformity", "age": 4, "level": -10602},
        {"name": "D3-D2d", "type": "conformable", "age": 3, "level": -11142},
        {"name": "D2d-D1y", "type": "conformable", "age": 2, "level": -11717},
        {"name": "D1y-D1n", "type": "conformable", "age": 1, "level": -12020}
    ]
}


class StratigraphicConstraint:
    """Stratigraphic Constraints Category"""
    def __init__(self, stratigraphic_column=STRATIGRAPHIC_COLUMN, min_level_value=-15000):
        self.strat_column = stratigraphic_column
        self.num_units = len(stratigraphic_column["units"])

        # Constructing a level-range lookup table(rock_unit -> (top_level, bottom_level))
        self.level_ranges = {}
        interface_dict = {ifc["age"]: ifc["level"] for ifc in stratigraphic_column["interfaces"]}

        for unit in stratigraphic_column["units"]:
            rock_unit = unit["rock_unit"]
            top_level = interface_dict[unit["top_interface"]]

            # Handling the lowest stratum: 'bottom_interface=0'indicates no bottom interface.
            if unit["bottom_interface"] == 0:
                # Employ a sufficiently deep virtual underworld
                bottom_level = min_level_value
            else:
                bottom_level = interface_dict[unit["bottom_interface"]]

            self.level_ranges[rock_unit] = (top_level, bottom_level)

        for rock_unit, (top, bottom) in sorted(self.level_ranges.items()):
            unit_name = next(u["name"] for u in stratigraphic_column["units"] if u["rock_unit"] == rock_unit)

    def get_level_ranges_tensor(self, device='cuda'):
        """
        Return the level range tensor
        Return: [num_units, 2] tensor, with the first column being bottom and the second column being top
        """
        ranges = torch.zeros(self.num_units, 2, device=device)
        for rock_unit, (top, bottom) in self.level_ranges.items():
            ranges[rock_unit - 1, 0] = bottom  
            ranges[rock_unit - 1, 1] = top 
        return ranges

    def compute_level_compatibility(self, predicted_levels, rock_unit_labels, device='cuda'):
        """
        Calculate the compatibility between predicted levels and lithological labels

        Parameters:
        - predicted_levels: [N] or [N, 1] predicted level value
        - rock_unit_labels: [N] Lithological label (1-based)

        RETURN:
        - compatibility: [N] Compatibility score, where 1 indicates full compatibility and 0 indicates incompatibility.
        """
        if predicted_levels.dim() > 1:
            predicted_levels = predicted_levels.squeeze()

        N = predicted_levels.shape[0]
        compatibility = torch.ones(N, device=device)

        for rock_unit in range(1, self.num_units + 1):
            mask = (rock_unit_labels == rock_unit)
            if mask.any():
                top_level, bottom_level = self.level_ranges[rock_unit]

                # Calculate deviation
                above_top = torch.clamp(predicted_levels[mask] - top_level, min=0)
                below_bottom = torch.clamp(bottom_level - predicted_levels[mask], min=0)
                deviation = above_top + below_bottom

                # Compatibility: The greater the deviation, the lower the compatibility.
                compatibility[mask] = torch.exp(-deviation / 1000.0)

        return compatibility

    def get_level_based_prior(self, predicted_levels, device='cuda', temperature=1000.0):
        """
        Calculating lithological prior probabilities based on level values

        Parameters:
        - predicted_levels: [N] or [N, 1] predicted level value
        - temperature: Temperature parameter, controlling the sharpness of the prior

        RETURN:
        - prior: [N, num_units] Prior probability distribution
        """
        # Ensure that predicted_levels is a 1D tensor.
        if predicted_levels.dim() > 1:
            predicted_levels = predicted_levels.squeeze()

        N = predicted_levels.shape[0]
        prior = torch.zeros(N, self.num_units, device=device)

        for rock_unit in range(1, self.num_units + 1):
            top_level, bottom_level = self.level_ranges[rock_unit]

            # Calculate the extent of level within the stratigraphic unit
            center = (top_level + bottom_level) / 2
            width = (top_level - bottom_level) / 2

            distance = torch.abs(predicted_levels - center)
            # Ensure that the assigned tensor is one-dimensional.
            prior[:, rock_unit - 1] = torch.exp(-distance ** 2 / (2 * (width / 2) ** 2))

        # Normalisation
        prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-8)

        return prior


class StratigraphicConstraintLoss(nn.Module):
    """Stratigraphic Constraint Loss Function"""

    def __init__(self, constraint_module, weight=1.0, temperature=1000.0):
        super().__init__()
        self.constraint = constraint_module
        self.weight = weight
        self.temperature = temperature

    def forward(self, predicted_levels, rock_unit_logits, rock_unit_labels, mask, device='cuda'):
        """
        Calculate formation-induced loss
        Parameters:
        - predicted_levels: [N] or [N, 1] predicted level value
        - rock_unit_logits: [N, num_classes] Lithological classification logits
        - rock_unit_labels: [N] True lithological label (1-based)
        - mask: [N] bool mask
        """
        if not mask.any():
            return torch.tensor(0.0, device=device)

        # Calculate only for labelled nodes
        pred_levels_masked = predicted_levels[mask]
        logits_masked = rock_unit_logits[mask]
        labels_masked = rock_unit_labels[mask]

        # Compute the level-based prior distribution
        level_prior = self.constraint.get_level_based_prior(
            pred_levels_masked, device=device, temperature=self.temperature
        )

        # Probability distribution predicted by computational models
        pred_probs = torch.softmax(logits_masked, dim=1)

        # KL divergence loss: Encourages predictions to align closely with the stratum prior
        kl_loss = torch.nn.functional.kl_div(
            pred_probs.log(),
            level_prior,
            reduction='batchmean'
        )

        # Add hard constraint: impose a penalty if the predicted lithology is entirely incompatible with the level.
        compatibility = self.constraint.compute_level_compatibility(
            pred_levels_masked, labels_masked, device=device
        )
        compatibility_loss = (1 - compatibility).mean()

        total_loss = kl_loss + compatibility_loss

        return self.weight * total_loss


class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha] * 13)
            else:
                if isinstance(alpha, torch.Tensor):
                    self.alpha = alpha.clone().detach()
                else:
                    self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(rock_unit_labels, mask, num_classes=13):
    """
    Calculate category weights
    """
    labels_masked = rock_unit_labels[mask].cpu().numpy()
    class_counts = np.bincount(labels_masked - 1, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts)
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)


def post_process_with_level_constraint(predicted_levels, rock_unit_logits,
                                       constraint_module, device='cuda'):
    # Ensure that predicted_levels is a 1D tensor.
    if predicted_levels.dim() > 1:
        predicted_levels = predicted_levels.squeeze()

    N = predicted_levels.shape[0]

    # Initial forecast
    original_preds = torch.argmax(rock_unit_logits, dim=1)

    # Calculate the compatibility of each node with all lithological units
    level_prior = constraint_module.get_level_based_prior(
        predicted_levels, device=device, temperature=500.0
    )

    # Model predicted probability
    pred_probs = torch.softmax(rock_unit_logits, dim=1)

    # Fusion of A Priori and Model Prediction
    alpha = 0.3  # A priori weights
    combined_probs = alpha * level_prior + (1 - alpha) * pred_probs

    # Make the final prediction based on the fused probability
    corrected_preds = torch.argmax(combined_probs, dim=1)
                                           
    return corrected_preds
