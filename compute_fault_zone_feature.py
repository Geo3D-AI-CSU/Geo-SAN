import torch
import numpy as np
if not hasattr(np,'bool'):
    np.bool = bool
import pyvista as pv
import os
import re
import glob
import vtk


def read_vtk_files(vtk_directory, pattern='F*.vtk'):
    """
    Read all .vtk files matching the specified pattern within the designated directory and extract their level values.

    Parameters:
    - vtk_directory (str): The directory path where .vtk files are stored.
    - pattern (str): File name matching pattern, default 'F*.vtk'.

    Return:
    - fault_meshes (dict): A dictionary with level as the key and PyVista.PolyData objects as the values.
    """
    vtk_files = glob.glob(os.path.join(vtk_directory, pattern))
    fault_meshes = {}
    for vtk_file in vtk_files:
        # Extract the level from the filename
        match = re.search(r'F(\d+)\.vtk', os.path.basename(vtk_file))
        if match:
            level = int(match.group(1))
            mesh = pv.read(vtk_file)
            fault_meshes[level] = mesh
        else:
            print(f"Non-conforming")
    return fault_meshes

def compute_fault_features(graph_data, vtk_directory,factor=1.0):
    """
    Generate fault characteristics, calculate the relationship between nodes and fault planes, and assign fault side attributes.

    Parameters:
    - graph_data (torch_geometric.data.Data):  The image data object must contain 'original_coords'.
    - vtk_directory (str): The directory path where .vtk files are stored.
    - visualize (bool): Is it possible to visualise the fracture plane?
    - output_dir (str): Directory for storing visualised images.

    return:
    - graph_data (torch_geometric.data.Data): The updated image data object incorporates fault characteristics.
    """
    # Extract node coordinates
    coords = graph_data.original_coords.cpu().numpy()
    num_nodes = coords.shape[0]

    # Read all vtk files for all fault planes
    fault_meshes = read_vtk_files(vtk_directory)

    num_levels = len(fault_meshes)
    fault_features = np.zeros((num_nodes, num_levels), dtype=int)

    # Create a global bounding box
    x_min, y_min, z_min = coords.min(axis=0) - 10.0  
    x_max, y_max, z_max = coords.max(axis=0) + 10.0
    global_bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

    # Traverse each fault plane, compute the signed distance and assign fault characteristics.
    for idx, (level, mesh) in enumerate(sorted(fault_meshes.items())):
        # Acquire normal vector data
        normals = mesh.GetPointData().GetNormals()
        try:
            if normals is None:
                mesh.compute_normals(inplace=True)

            # Convert the coordinates of the image data to VTK format
            vtk_points = vtk.vtkPoints()
            for coord in coords:  # coords should be the node coordinates in the graph data
                vtk_points.InsertNextPoint(coord)

            # Create a VTK PolyData object and assign the node coordinates to it.
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(vtk_points)

            # Convert VTK PolyData to PyVista PolyData
            pyvista_poly_data = pv.wrap(poly_data)

            # Using PyVista to wrap fracture surface meshes
            pyvista_mesh = pv.wrap(mesh)

            # Calculate the distance from the computational node to the fault plane
            result = pyvista_poly_data.compute_implicit_distance(pyvista_mesh,inplace=True)
            distances = result['implicit_distance']
            if level == 1:
                distances = -distances   
            # Determine whether a node is in the upper or lower half based on the sign of the distance.
            fault_sides = (distances > 0).astype(int)  # Distance > 0 is 1 (upper deck), <= 0 is 0 (lower deck)
            fault_features[:, idx] = fault_sides
        except Exception as e:
            print(f" Level {level}..... ")

    # Converting fault characteristics into tensors and transferring them to the device
    fault_features_tensor = torch.tensor(fault_features, dtype=torch.float32).to(graph_data.x.device)
    # Append fault characteristics to existing node features
    graph_data.x = torch.cat([graph_data.x, fault_features_tensor], dim=-1)
    return graph_data
