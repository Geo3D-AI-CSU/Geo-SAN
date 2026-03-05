import torch
import torch.nn.functional as F


def scalar_loss_slow(predicted_scalar, rock_unit_labels, min_values, max_values):
    """
    Compute the loss function for the lithological scalar field, 
    ensuring predicted scalar values remain within the specified range, 
    whilst handling missing minimum and maximum values.

    Parameters：
    - predicted_scalar (Tensor): Predicted scalar field values, in the shape of (num_nodes, )
    - rock_unit_labels (Tensor): Lithological tag, of shape (num_nodes, ), used to locate the corresponding range.
    - min_values (Tensor or None): The minimum scalar field value corresponding to each lithological label, shaped as (num_labels, ), may be None.
    - max_values (Tensor or None): The maximum scalar field value corresponding to each lithological label, shaped as (num_labels, ), may be None.

   RETURN：
    - loss (Tensor): The calculated loss value.
    """

    # Obtain the minimum and maximum values corresponding to the lithological labels
    min_values_for_labels = min_values[rock_unit_labels] if min_values != -9999 else -9999
    max_values_for_labels = max_values[rock_unit_labels] if max_values != -9999 else -9999

    # Ensure that min_values and max_values align with predicted_scalar on the device.
    if min_values_for_labels is not -9999:
        min_values_for_labels = min_values_for_labels.to(predicted_scalar.device)
    if max_values_for_labels is not -9999:
        max_values_for_labels = max_values_for_labels.to(predicted_scalar.device)

    # Initialise loss value
    loss = torch.zeros_like(predicted_scalar,requires_grad=True)

    # Iterate through each node and compute the loss
    for i in range(len(predicted_scalar)):
        min_value = min_values_for_labels[i]
        max_value = max_values_for_labels[i]
        scalar_value = predicted_scalar[i]

        # Determine the values of min_value and max_value and calculate the loss.
        if min_value == -9999 and max_value == -9999:
            continue 
        if min_value == -9999:
            loss = torch.where(scalar_value > max_value, torch.abs(scalar_value - max_value), loss)
        elif max_value == -9999:
            loss = torch.where(scalar_value < min_value, torch.abs(scalar_value - min_value), loss)
        else:
            loss = torch.where(scalar_value > max_value, torch.abs(scalar_value - max_value), loss)
            loss = torch.where(scalar_value < min_value, torch.abs(scalar_value - min_value), loss)

    return loss.sum() 


def scalar_loss(predicted_scalar, rock_unit_labels, min_values, max_values):
    """
    Compute the loss function for the lithological scalar field,
    ensuring predicted scalar values remain within the specified range, 
    whilst handling missing minimum and maximum values.

    Parameters：
    - predicted_scalar (Tensor):  The predicted scalar field values, in the shape of (num_nodes, ).
    - rock_unit_labels (Tensor): Lithological tag, of shape (num_nodes, ), used to locate the corresponding range.
    - min_values (Tensor or None): The minimum scalar field value corresponding to each lithological label, shaped as (num_labels, ), may be None.
    - max_values (Tensor or None): The maximum scalar field value corresponding to each lithological label, shaped as (num_labels, ), may be None.

    RETURN：
    - loss (Tensor): The calculated loss value.
    """

    # Obtain the minimum and maximum values corresponding to the lithological labels
    min_values_for_labels = min_values[rock_unit_labels-1]  # Obtain the minimum value corresponding to each lithological label
    max_values_for_labels = max_values[rock_unit_labels-1]  # Obtain the maximum value corresponding to each lithological label

    # Ensure that min_values and max_values align with predicted_scalar on the device.
    min_values_for_labels = min_values_for_labels.to(predicted_scalar.device)
    max_values_for_labels = max_values_for_labels.to(predicted_scalar.device)

    # Initialise loss value
    loss = torch.zeros_like(predicted_scalar, requires_grad=True)

    # Calculate the loss of scalar values
    min_mask = min_values_for_labels != -9999
    max_mask = max_values_for_labels != -9999

    loss = torch.where(~min_mask & max_mask & (predicted_scalar > max_values_for_labels),
                        torch.abs(predicted_scalar - max_values_for_labels),
                        loss)

    loss = torch.where(~max_mask & min_mask & (predicted_scalar < min_values_for_labels),
                        torch.abs(predicted_scalar - min_values_for_labels),
                        loss)

    loss = torch.where(min_mask & max_mask & (predicted_scalar > max_values_for_labels),
                        torch.abs(predicted_scalar - max_values_for_labels),
                        loss)
    loss = torch.where(min_mask & max_mask & (predicted_scalar < min_values_for_labels),
                        torch.abs(predicted_scalar - min_values_for_labels),
                        loss)
    # The sum of the loss values for all nodes
    return loss.sum()


# Define the loss function
def level_loss(predicted, target):
    return  F.mse_loss(predicted, target)

def rock_unit_loss(predicted_rock_units, rock_unit_labels):
    rock_unit_labels = rock_unit_labels - 1
    return F.cross_entropy(predicted_rock_units, rock_unit_labels)

def gradient_loss(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    """
    Compute the gradient loss.
    :param predicted_levels: Predicted level value
    :param coords: Node coordinates
    :param dx, dy, dz: True gradient
    :param edge_index: Edge index of the graph
    :param mask_gradient: Mask, specifying which nodes require gradient computation
    :return: Gradient loss
    mask_sample: Used to indicate which nodes require gradient computation; only nodes with neighbouring nodes shall compute gradients.
    """
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    mask_sample = torch.zeros(mask_indices .size(0), dtype=torch.bool)  
    row, col = edge_index
    gradients = []

    # Precompute the Boolean array for row == node and col == node to minimise redundant calculations.
    for i, node in enumerate(mask_indices):
        row_mask = (row == node) 
        col_mask = (col == node)  

        # Identify the neighbours within row and col that are associated with the current node.
        col_neighbors = col[row_mask]  
        row_neighbors = row[col_mask]

        # Merge neighbouring rows and columns
        neighbors = torch.cat((row_neighbors, col_neighbors))

        # Update the mask, check for neighbours
        if neighbors.numel() > 0:
            mask_sample[i] = True  
        else:
            mask_sample[i] = False  

        # Skip nodes with no neighbours
        if not mask_sample[i]:
            gradients.append(torch.zeros(3, device=predicted_levels.device)) 
            continue

        neighbors_v = neighbors
        # Calculate delta_coords and delta_levels
        delta_coords = coords[neighbors_v] - coords[node].unsqueeze(0)
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[node].unsqueeze(0)

        # Compute gradients
        delta_coords.requires_grad_(True)
        delta_levels.requires_grad_(True)

        # Compute the transpose of the Jacobian matrix and multiply it by delta_coords
        AtA = torch.matmul(delta_coords.T, delta_coords)
        Atb = torch.matmul(delta_coords.T, delta_levels.unsqueeze(1))

        # Use torch.linalg.pinv for stable solution
        try:
            gradient = torch.linalg.pinv(AtA) @ Atb
            gradients.append(gradient.squeeze())
        except:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            print(f"Solving fail, setting gradient to [0, 0, 0].")

    # After calculating the gradients for all nodes, stack them into a tensor.
    gradient_estimates = torch.stack(gradients) if gradients else torch.zeros_like(dx)  

    # True gradient
    true_gradients = torch.stack([dx, dy, dz], dim=-1)

    # Normalised gradient
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients
    mask_sample = mask_sample.to(normalized_true_gradients.device)
    # Calculate the gradient loss under the mask
    masked_predicted_gradients = normalized_predicted_gradients * mask_sample.unsqueeze(-1)  
    masked_true_gradients = normalized_true_gradients * mask_sample.unsqueeze(-1)  

    # Calculate the cosine similarity between the predicted gradient and the actual gradient
    cos_theta = (masked_predicted_gradients * masked_true_gradients).sum(dim=-1)
    mask = (cos_theta != 0)
    # Calculate angle loss
    angle_loss = torch.mean(1 - cos_theta[mask])

    return angle_loss


def gradient_loss_slow(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    """
    Compute the gradient loss.
    """
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    mask_sample = torch.zeros(mask_indices .size(0), dtype=torch.bool) 
    row, col = edge_index
    gradients = []

    # Iterate through each node, checking whether it has neighbours; if so, set it to True.
    for i, node in enumerate(mask_indices):
        # Check whether the node is in the row or column
        if (row == node).any() and (col == node).any():
            neighbors = torch.cat((col[row == node], row[col == node])) 
        elif (row == node).any():
            neighbors = col[row == node]
        elif (col == node).any():
            neighbors = row[col == node]
        if neighbors.numel() > 0:
            mask_sample[i] = True  
        else:
            mask_sample[i] = False  
        if not mask_sample[i]: 
            # print(f"Node {v} has no neighbors, setting gradient to [0, 0, 0].")
            gradients.append(torch.zeros(3, device=predicted_levels.device))  
            continue  
        neighbors_v = neighbors
        delta_coords = coords[neighbors_v] - coords[node].unsqueeze(0)
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[node].unsqueeze(0)
        delta_coords.requires_grad_(True)
        delta_levels.requires_grad_(True)
        AtA = torch.matmul(delta_coords.T, delta_coords)
        Atb = torch.matmul(delta_coords.T, delta_levels.unsqueeze(1))
        try:
            gradient = torch.linalg.pinv(AtA) @ Atb
            gradients.append(gradient.squeeze())
        except:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            print(f"Solving fail, setting gradient to [0, 0, 0].")

    # After calculating the gradients for all nodes, stack them into a single tensor.
    gradient_estimates = torch.stack(gradients) if gradients else torch.zeros_like(dx) 

    # True gradient
    true_gradients = torch.stack([dx, dy, dz], dim=-1)

    # Normalised gradient
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients
    mask_sample = mask_sample.to(normalized_true_gradients.device)
    # Calculate the gradient loss under the mask
    masked_predicted_gradients = normalized_predicted_gradients * mask_sample.unsqueeze(-1)  
    masked_true_gradients = normalized_true_gradients * mask_sample.unsqueeze(-1)  

    # Calculate the cosine similarity between the predicted gradient and the actual gradient
    cos_theta = (masked_predicted_gradients * masked_true_gradients).sum(dim=-1)
    mask = (cos_theta != 0)
    # Calculate angle loss
    angle_loss = torch.sum(1 - cos_theta[mask])

    return angle_loss


def gradient_loss_old(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    # Obtain the node indices for which gradients need to be computed
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    num_masked = mask_indices.size(0)

    # The source and destination of the separation edge
    row, col = edge_index

    # Prepare a list to store the gradient for each node.
    gradients = []

    for v in mask_indices:
        # Obtain the neighbouring nodes of node v
        neighbors_v = col[row == v]

        if neighbors_v.numel() == 0:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            continue

        # Calculate the difference between coordinates and level
        delta_coords = coords[neighbors_v] - coords[v].unsqueeze(0)  # [num_neighbors, 3]
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[v].unsqueeze(0)  # [num_neighbors]

        # Solving the gradient using the method of least squares
        AtA = torch.matmul(delta_coords.T, delta_coords)  # [3, 3]
        Atb = torch.matmul(delta_coords.T, delta_levels.unsqueeze(1))  # [3, 1]

        # Compute gradients
        try:
            # Using pseudoinverse to avoid non-positive definite matrix issues
            gradient = torch.linalg.pinv(AtA) @ Atb  # [3, 1]
            gradients.append(gradient.squeeze())
        except:
            gradients.append(torch.zeros(3, device=predicted_levels.device))

    # Merge all gradients
    gradient_estimates = torch.stack(gradients)  # [num_masked, 3]

    # Compute the true gradient
    true_gradients = torch.stack((dx, dy, dz), dim=-1)  # [num_masked, 3]

    # Normalised gradient
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients

    # Calculate the cosine similarity between the predicted gradient and the actual gradient
    cos_theta = (normalized_predicted_gradients * normalized_true_gradients).sum(dim=-1)
    angle_loss = torch.sum(1 - cos_theta)

    return angle_loss



def gradient_loss_autogradient(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    """
    Compute the gradient at the coordinates corresponding to the predicted 'level', 
    and compare it with the target gradient (dx, dy, dz) to calculate the angular loss.
    """
    # Positions where mask_gradient is set to True require gradient computation.
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    row, col = edge_index  # Obtain the connection information for the edges

    gradients = []

    # Traverse all nodes requiring gradient computation
    for v in mask_indices:
        neighbors_v = col[row == v]
        if neighbors_v.numel() == 0:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            continue

        # Calculate delta_coords and delta_levels
        delta_coords = coords[neighbors_v] - coords[v].unsqueeze(0)
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[v].unsqueeze(0)

        # Gradients computed using automatic differentiation for prediction
        coords_v = coords[v].unsqueeze(0).requires_grad_(True) 
        predicted_levels_v = predicted_levels[v].unsqueeze(0).requires_grad_(True) 

        # Calculate the loss
        loss = torch.abs(predicted_levels_v - predicted_levels[neighbors_v])

        # Clear the gradient to avoid it being affected when calculating other gradients.
        coords_v.grad = None  

        # Backpropagation, calculating gradients
        loss.backward()

        # Obtain the gradient for node v
        gradient = coords_v.grad
        gradients.append(gradient.squeeze())


    gradient_estimates = torch.stack(gradients)
    true_gradients = torch.stack([dx, dy, dz], dim=-1)
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients
    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients

    # Calculate the cosine similarity between the predicted gradient and the actual gradient
    cos_theta = (normalized_predicted_gradients * normalized_true_gradients).sum(dim=-1)

    # Calculate angle loss
    angle_loss = torch.sum(1 - cos_theta)

    return angle_loss
