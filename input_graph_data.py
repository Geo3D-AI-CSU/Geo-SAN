import numpy as np
import torch
from numpy.lib.function_base import gradient
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from itertools import combinations
import os
from maths import  azimuthplunge2vector,strikedip2vector


# Loading node data
def load_node_data_beifen(node_file,is_gradient=False):
    if is_gradient == False:
        with open(node_file, 'r') as f:
            header = f.readline()
            n_nodes = int(header.strip().split()[0])
        node_df = pd.read_csv(
            node_file,
            delim_whitespace=True,
            skiprows=1,
            nrows=n_nodes,
            header=None,
            engine='c',
            dtype={0: np.int32, 1: np.float32, 2: np.float32, 3: np.float32,
                   4: np.float32, 5: np.float32, 6: np.float32, 7: np.float32, 8: str}  
        )
        node_data = node_df.values
    else:
        with open(node_file, 'r') as f:
            header = f.readline()
            n_nodes = int(header.strip().split()[0])
        node_df = pd.read_csv(
            node_file,
            delim_whitespace=True,
            skiprows=1,
            nrows=n_nodes,
            header=None,
            engine='c',
            dtype={0: np.int32, 1: np.float32, 2: np.float32, 3: np.float32,
                   4: np.float32, 5: np.float32, 6: np.float32, 7: np.float32, 8: np.float32, 9: str,}  
        )
        node_data = node_df.values
    return node_data

def load_node_data(node_file):
    with open(node_file, 'r') as f:
        header = f.readline()
        n_nodes = int(header.strip().split()[0])
    node_df = pd.read_csv(
        node_file,
        delim_whitespace=True,
        skiprows=1,
        nrows=n_nodes,
        header=None,
        engine='c',
        dtype=np.float32
    )
    node_data = node_df.values
    return node_data


# Loading edge data
def load_edge_data_beifen(ele_file):
    with open(ele_file, 'r') as f:
        header = f.readline()
    edge_df = pd.read_csv(
        ele_file,
        delim_whitespace=True,
        skiprows=1,
        header=None,
        usecols=[1, 2, 3, 4],
        engine='c',
        dtype=np.int32,
        on_bad_lines='skip'
    )
    edge_data = edge_df.values # Indexing converted from starting at 1 to starting at 0
    # edge_data = edge_df.values - 1  
    return edge_data
def load_edge_data(ele_file):
    with open(ele_file, 'r') as f:
        header = f.readline()
    edge_df = pd.read_csv(
        ele_file,
        delim_whitespace=True,
        skiprows=1,
        header=None,
        usecols=[1, 2, 3, 4],
        engine='c',
        dtype=np.int32,
        on_bad_lines='skip'
    )
    edge_data = edge_df.values
    return edge_data


def create_graph(node_data, ele_data,is_gradient=False):
    if is_gradient == False:
        coords = node_data[:, 1:4]  # Extract x, y, z
        properties = node_data[:, 4:]  # Other attributes: rock_unit, QJ, QX, level

        # Extract tag data
        level = properties[:, -1]  # level (iso-value fv)
        QJ = properties[:, 1]  # QJ
        QX = properties[:,2]   # QX
        gradient = strikedip2vector(QJ, QX)
        dx = gradient[:,0].astype(np.float32)
        dy = gradient[:,1].astype(np.float32)
        dz = gradient[:,2].astype(np.float32)
        rock_unit = properties[:, 0]  # rock_unit
        #  Create mask
        mask_level = (~np.isnan(level)) & (level != -9999)
        mask_rock_unit = (~np.isnan(rock_unit)) & (rock_unit != -9999)
        mask_gradient = (~np.isnan(QJ)) & (QJ != -9999) & (~np.isnan(QX)) & (QX != -9999)

    else:
        coords = node_data[:, 1:4]  
        properties = node_data[:, 4:] 
        level = properties[:, -1]  # level (iso-value fv)
        dx = properties[:, 1]  # dx
        dy = properties[:, 2]  # dy
        dz = properties[:, 3]  # dz
        rock_unit = properties[:, 0]  # rock_unit 
        mask_level = (~np.isnan(level)) & (level != -9999)
        mask_rock_unit = (~np.isnan(rock_unit)) & (rock_unit != -9999)
        mask_gradient = (~np.isnan(dx)) & (dx != -9999) & (~np.isnan(dy)) & (dy != -9999) & (~np.isnan(dz)) & (dz != -9999)

    edge_index = []
    edge_set = set()
    for tetra in ele_data:
        for u, v in combinations(tetra, 2):
            # Implementing deduplication for ordered tuples through standardisation
            sorted_edge = tuple(sorted((u, v)))
            edge_set.add(sorted_edge)

    # Convert to zero-based PyTorch format
    edge_index = torch.tensor(
        [[u - 1, v - 1] for u, v in edge_set],  #  Assuming the original node numbering starts from 1
        dtype=torch.long
    ).t().contiguous()

    # Normalised coordinates
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_coords = scaler.fit_transform(coords)
    np.save('normalization_params.npy', {'min': scaler.data_min_, 'range': scaler.data_range_})

    # Node features comprise solely normalised (x, y, z) coordinates.
    node_features = torch.tensor(normalized_coords, dtype=torch.float)

    # Create graph data objects
    graph_data = Data(
        x=node_features,  # Node characteristics (x, y, z)
        edge_index=edge_index,
        mask_rock_unit=torch.tensor(mask_rock_unit, dtype=torch.bool),
        mask_level=torch.tensor(mask_level, dtype=torch.bool),
        mask_gradient=torch.tensor(mask_gradient, dtype=torch.bool),
        level=torch.tensor(level, dtype=torch.float),  # level label (iso-value)
        rock_unit=torch.tensor(rock_unit, dtype=torch.long),  # rock_unit 
        gradient=torch.tensor(np.stack((dx, dy, dz), axis=-1), dtype=torch.float),  # dx, dy, dz
        original_coords=torch.tensor(coords, dtype=torch.float)
    )
    print("Graph Data Attributes after creation:", list(graph_data.keys))
    print("Level Attribute Shape:", graph_data.level.shape)
    print("Sample Level Data:", graph_data.level[:5])
    return graph_data


def create_graph_old_edge(node_data, ele_data,is_gradient=False):
    if is_gradient == False:
        coords = node_data[:, 1:4]  
        properties = node_data[:, 4:]  
        level = properties[:, -1]  # level (iso-value fv)
        QJ = properties[:, 1]  # QJ
        QX = properties[:,2]   # QX
        gradient = strikedip2vector(QJ, QX)
        dx = gradient[:,0].astype(np.float32)
        dy = gradient[:,1].astype(np.float32)
        dz = gradient[:,2].astype(np.float32)
        rock_unit = properties[:, 0]  # rock_unit 

    else:
        coords = node_data[:, 1:4]  #  x, y, z
        properties = node_data[:, 4:]  # rock_unit, dx, dy, dz, level
        level = properties[:, -1]  # level (iso-value fv)
        dx = properties[:, 1]  # dx
        dy = properties[:, 2]  # dy
        dz = properties[:, 3]  # dz
        rock_unit = properties[:, 0]  # rock_unit 

    # Create mask
    mask_level = (~np.isnan(level)) & (level != -9999)
    mask_rock_unit = (~np.isnan(rock_unit)) & (rock_unit != -9999)
    mask_gradient = (~np.isnan(dx)) & (dx != -9999) & (~np.isnan(dy)) & (dy != -9999) & (~np.isnan(dz)) & (dz != -9999)

    # Create edge indexes and remove duplicate edges
    edge_list = []
    for tetra in ele_data:
        # Use combinations to obtain all edges, pairing each node (u, v)
        edges = list(combinations(tetra, 2))  # [(n1, n2), (n1, n3), ...]
        # Sort node indices to ensure (min, max)
        sorted_edges = [tuple(sorted(edge)) for edge in edges]
        edge_list.extend(sorted_edges)
    # Convert to a NumPy array
    edge_array = np.array(edge_list, dtype=np.int32)
    # Remove duplicate edges
    edge_unique = np.unique(edge_array, axis=0)
    # Transpose into (2, num_edges) and convert to a torch tensor
    edge_index = torch.tensor(edge_unique.T, dtype=torch.long)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_coords = scaler.fit_transform(coords)
    np.save('normalization_params.npy', {'min': scaler.data_min_, 'range': scaler.data_range_})

    node_features = torch.tensor(normalized_coords, dtype=torch.float)
    # Create graph data objects
    graph_data = Data(
        x=node_features,  # x, y, z
        edge_index=edge_index,
        mask_rock_unit=torch.tensor(mask_rock_unit, dtype=torch.bool),
        mask_level=torch.tensor(mask_level, dtype=torch.bool),
        mask_gradient=torch.tensor(mask_gradient, dtype=torch.bool),
        level=torch.tensor(level, dtype=torch.float),  # level 
        rock_unit=torch.tensor(rock_unit, dtype=torch.long),  # rock_unit 
        gradient=torch.tensor(np.stack((dx, dy, dz), axis=-1), dtype=torch.float),  # dx, dy, dz
        original_coords=torch.tensor(coords, dtype=torch.float)
    )

    print("Graph Data Attributes after creation:", list(graph_data.keys))
    print("Level Attribute Shape:", graph_data.level.shape)
    print("Sample Level Data:", graph_data.level[:5])

    return graph_data

def create_graph_beifen(node_data, ele_data,is_gradient=False):
    if is_gradient == False:
        coords = node_data[:, 1:4].astype(np.float32)  #  x, y, z
        properties = node_data[:, 4:-1].astype(np.float32)  # Level,Rock_unit,QJ,QX,Attribute

        attribute = node_data[:, -1]
        rock_unit = properties[:, 1]
        QJ = properties[:, 2]
        QX = properties[:, 3]
        level= properties[:, 0]
        gradient = strikedip2vector(QJ, QX)
        dx =  gradient[:, 0].astype(np.float32)
        dy =  gradient[:, 1].astype(np.float32)
        dz =  gradient[:, 2].astype(np.float32)
    else:
        coords = node_data[:, 1:4].astype(np.float32)  # x, y, z
        properties = node_data[:, 4:-1].astype(np.float32)  # Level,Rock_unit,dx,dy,dz,Attribute
        attribute = node_data[:, -1]  
        rock_unit = properties[:, 1]
        level = properties[:, 0]
        dx = properties[:, 2]
        dy = properties[:, 3]
        dz = properties[:, 4]
    mask_level = (level != -9999)
    mask_rock_unit = (rock_unit != -9999)
    mask_gradient = (~np.isnan(dx)) & (dx != -9999) & (~np.isnan(dy)) & (dy != -9999) & (~np.isnan(dz)) & (dz != -9999)


    edge_list = []
    for tetra in ele_data:
        edges = list(combinations(tetra, 2))  # [(n1, n2), (n1, n3), ...]
        sorted_edges = [tuple(sorted(edge)) for edge in edges]
        edge_list.extend(sorted_edges)
    edge_array = np.array(edge_list, dtype=np.int32)
    edge_unique = np.unique(edge_array, axis=0)
    edge_index = torch.tensor(edge_unique.T, dtype=torch.long)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_coords = scaler.fit_transform(coords)
    node_features = torch.tensor(normalized_coords, dtype=torch.float)
    graph_data = Data(
        x=node_features, 
        edge_index=edge_index,
        mask_rock_unit=torch.tensor(mask_rock_unit, dtype=torch.bool),
        mask_level=torch.tensor(mask_level, dtype=torch.bool),
        mask_gradient=torch.tensor(mask_gradient, dtype=torch.bool),
        level=torch.tensor(level, dtype=torch.float), 
        rock_unit=torch.tensor(rock_unit, dtype=torch.long), 
        gradient=torch.tensor(np.stack((dx, dy, dz), axis=-1), dtype=torch.float), 
        original_coords=torch.tensor(coords, dtype=torch.float), 
        attribute=attribute
    )
    print("Graph Data Attributes after creation:", list(graph_data.keys))
    print("Level Attribute Shape:", graph_data.level.shape)
    print("Sample Level Data:", graph_data.level[:5])

    return graph_data


# Load or create graph data
def create_or_load_graph(node_file, ele_file, pt_file=None,is_gradient=False):
    """
    Check whether a saved graph data PT file exists. If present, load it directly; otherwise, create the graph structure data and save it.
    """
    if pt_file is None:
        node_dir = os.path.dirname(node_file)
        pt_file = os.path.join(node_dir, 'graph_data.pt')
    if os.path.exists(pt_file):
        graph_data = torch.load(pt_file)
    else:
        node_data = load_node_data(node_file)
        edge_data = load_edge_data(ele_file)
        graph_data = create_graph(node_data, edge_data,is_gradient=is_gradient)
        torch.save(graph_data, pt_file)
    return graph_data
