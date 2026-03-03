# Overview
The current three-dimensional (3D) geological implicit modelling methods are mainly based on interpolation methods, such as Kriging and radial basis functions (RBFs), which struggle to capture the nonlinear characteristics of complex geological structures and are limited in their capacity to integrate multi-source modeling data. To overcome these limitations, we proposed a 3D geological modelling framework, Geo-SAN, which consists of a dual-task stratigraphy-aware attention network. The framework starts with graph neural networks (GNNs) with a multi-scale neighborhood aggregation mechanism which is aimed to identify critical sampled points adjacent to fault planes and aggregate the lithological features. Subsequently, a stratigraphy-aware attention mechanism is introduced to explicitly incorporate similarities in stratigraphic sequence into the framework. A unidirectional stratigraphic scalar field penalty to lithological classification is developed and incorporated into loss functions, thereby denoising lithological classification. Finally, a dual-task prediction head is designed to simultaneously complete lithological classification and scalar field interpolation. Ablation experiment further validates the contributions of the three core components, that is, graph neighborhood aggregation, stratigraphy-aware attention, and dual-task learning.
# Installtion
## Prerequisites
Python3.11+  
PyTorch 2.2+  
CUDA 12.1 (for GPU acceleration)  
# Data Preparation
## Required Data
1. DEM
2. Fault(VTK)
3. Boundary(Modeling area)
4. Sampaling data(interface,section)
# Running the Project
## Step 1: Build a knowledge graph
The tetrahedral meshing of the sampled points is performed to create graph structured data. All the sampling points and the other densifying points are both included into the graph data. 
```python
python tetra_partition_dem_control_scipy_delaunay.py --output graph_data.pt
```
## Step 2: Build a knowledge graph
