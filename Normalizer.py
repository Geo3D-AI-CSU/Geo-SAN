import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Normalizer:
    def __init__(self):
        self.level_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.coord_scaler = MinMaxScaler(feature_range=(-1, 1))

    def fit_transform_level_masked(self, level_masked):
        """
        Normalise only the level of the masked node and fit the scaler.
        """
        level_np = level_masked.cpu().numpy().reshape(-1, 1)
        self.level_scaler.fit(level_np)
        level_norm = self.level_scaler.transform(level_np)
        return torch.tensor(level_norm.squeeze(), dtype=torch.float32).to(
            level_masked.device)  # Modify: Use squeeze() to make it a one-dimensional tensor

    def inverse_transform_level(self, level_norm):
        """
        1. Add .detach() to prevent gradient errors.
        2. Use .squeeze() to ensure a 1D tensor is returned.
        """
        level_np = level_norm.detach().cpu().numpy().reshape(-1, 1)
        level_original = self.level_scaler.inverse_transform(level_np)
        # Return a 1D tensor to avoid dimension mismatches.
        return torch.tensor(level_original.squeeze(), dtype=torch.float32).to(level_norm.device)

    def fit_transform_coords(self, coords):
        coords_np = coords.cpu().numpy()
        coords_norm = self.coord_scaler.fit_transform(coords_np)
        return torch.tensor(coords_norm, dtype=torch.float32).to(coords.device)

    def inverse_transform_coords(self, coords_norm):
        coords_np = coords_norm.cpu().numpy()
        coords_original = self.coord_scaler.inverse_transform(coords_np)
        return torch.tensor(coords_original, dtype=torch.float32).to(coords_norm.device)

    def fit_transform_values(self, min_values, max_values):
        """
        Normalise min_values and max_values to the interval [0, 1].
        Ignore values of -9999 and do not normalise them.
        """
        # Convert to a NumPy array for processing
        min_values = np.array(min_values)
        max_values = np.array(max_values)

        # Normalisation processing, utilising valid data (ignoring values of -9999)
        valid_mask_min = min_values != -9999  # Mark valid min_values
        valid_mask_max = max_values != -9999  # Mark valid max_values

        # Normalise min_values and max_values, normalising only valid values.
        min_values_norm = np.copy(min_values)
        max_values_norm = np.copy(max_values)
        max_values_norm = max_values_norm.astype(np.float64)
        min_values_norm = min_values_norm.astype(np.float64)
        # Normalise only the effective value portion
        if np.any(valid_mask_min):  # Ensure the effective minimum value portion is present
            valid_min_values = min_values[valid_mask_min]  # Obtain the valid portion
            min_norm = (valid_min_values - valid_min_values.min()) / (
                        valid_min_values.max() - valid_min_values.min())  # Normalised effective portion
            min_values_norm[valid_mask_min] = min_norm  # Return the normalised result as an assigned value.

        if np.any(valid_mask_max):  # Ensure the presence of the effective maximum value section
            valid_max_values = max_values[valid_mask_max]  # Obtain the valid portion
            max_norm = (valid_max_values - valid_max_values.min()) / (
                        valid_max_values.max() - valid_max_values.min())  # Normalised effective portion
            max_values_norm[valid_mask_max] = max_norm  # Return the normalised result as an assigned value.

        return min_values_norm, max_values_norm

    def inverse_transform_values(self, min_values_norm, max_values_norm, min_values, max_values):
        min_values_norm = np.array(min_values_norm)
        max_values_norm = np.array(max_values_norm)
        min_values = np.array(min_values)
        max_values = np.array(max_values)

        valid_mask = (min_values_norm != -9999) & (max_values_norm != -9999)

        min_values_original = np.copy(min_values_norm)
        max_values_original = np.copy(max_values_norm)

        min_values_original[valid_mask] = min_values_norm[valid_mask] * (max_values[valid_mask] - min_values[valid_mask]) + min_values[valid_mask]
        max_values_original[valid_mask] = max_values_norm[valid_mask] * (max_values[valid_mask] - min_values[valid_mask]) + min_values[valid_mask]

        return min_values_original, max_values_original
