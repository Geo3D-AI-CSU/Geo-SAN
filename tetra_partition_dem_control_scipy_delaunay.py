from shapely.geometry import Polygon, Point
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree
import os
import time

from sqlalchemy.sql.functions import current_time
from tqdm import tqdm
import pyvista as pv
import matplotlib.pyplot as plt
import datetime

def filter_sampling_points_by_polygon_and_z_range(sampling_points, polygon, z_range):
    """Filter sampling points based on boundary polygons and Z-range criteria, 
    retaining only those points situated within the polygons and meeting the specified Z-value requirements."""
    # First filter the Z-values
    mask_z = (sampling_points[:, 2] >= z_range[0]) & (sampling_points[:, 2] <= z_range[1])
    points_z_filtered = sampling_points[mask_z]

    if len(points_z_filtered) == 0:
        return points_z_filtered, mask_z

    # Polygon filtering
    xy_points = [Point(point[0], point[1]) for point in points_z_filtered]
    mask_polygon = np.array([polygon.contains(pt) for pt in xy_points], dtype=bool)

    # Update mask
    final_mask = np.zeros_like(mask_z, dtype=bool)
    final_mask[mask_z] = mask_polygon

    return points_z_filtered[mask_polygon], final_mask


def load_boundary_points(boundary_csv):
    """Load the X, Y, Z data for the split boundary points"""
    df_boundary = pd.read_csv(boundary_csv)
    boundary_points = df_boundary[['X', 'Y', 'Z']].values
    return boundary_points


def load_dem_data(dem_csv):
    """Loading DEM data"""
    df_dem = pd.read_csv(dem_csv)
    dem_points = df_dem[['X', 'Y', 'Z']].values
    return dem_points


def is_point_in_polygon(point, polygon):
    """Check whether the point lies within the polygon"""
    return polygon.contains(Point(point[0], point[1])) or polygon.intersects(Point(point[0], point[1]))


def is_points_batch_in_polygon(points_batch, polygon):
    """Batch check whether points lie within polygons"""
    return np.array([polygon.contains(Point(pt[0], pt[1])) or
                     polygon.intersects(Point(pt[0], pt[1])) for pt in points_batch])


def generate_grid_points_below_dem(dem_points, z_min, spacing_x, spacing_y, spacing_z, polygon):
    """Optimised grid point generation algorithm utilising batch processing and pre-allocated memory"""

    # Obtain the XY coordinates of the DEM
    x_min, x_max = polygon.bounds[0], polygon.bounds[2]
    y_min, y_max = polygon.bounds[1], polygon.bounds[3]

    # Generate X and Y ranges using different spacing
    x_range = np.arange(x_min, x_max + spacing_x, spacing_x)
    y_range = np.arange(y_min, y_max + spacing_y, spacing_y)

    # Create an XY grid
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    xy_points = np.vstack([X.ravel(), Y.ravel()]).T

    # Batch check whether points lie within polygons
    batch_size = 10000  
    in_polygon_mask = np.zeros(len(xy_points), dtype=bool)

    for i in tqdm(range(0, len(xy_points), batch_size), desc="Polygon inspection"):
        batch = xy_points[i:i + batch_size]
        in_polygon_mask[i:i + batch_size] = is_points_batch_in_polygon(batch, polygon)

    # Only retain points within the polygon
    xy_points_filtered = xy_points[in_polygon_mask]


    # The KDTree used for constructing DEMs facilitates rapid lookup operations.
    dem_tree = cKDTree(dem_points[:, :2])

    # Query Z-values on the DEM
    dist, indices = dem_tree.query(xy_points_filtered, k=1)
    dem_z_values = dem_points[indices, 2]

    # Calculate the number of Z-values to be generated for each point
    z_points_per_xy = np.ceil((dem_z_values - z_min) / spacing_z).astype(int) + 1
    total_points = np.sum(z_points_per_xy)

    # Allocate the final point array
    grid_points = np.zeros((total_points, 3))

    # Fill the point array
    current_index = 0
    for i, (x, y, top_z, num_z) in enumerate(zip(xy_points_filtered[:, 0],
                                                 xy_points_filtered[:, 1],
                                                 dem_z_values,
                                                 z_points_per_xy)):
        # Generate all z-values for this xy position
        z_values = np.linspace(top_z, z_min, num_z)

        # Calculate the indices of these points in the final array
        end_index = current_index + len(z_values)

        # Fill XYZ coordinates
        grid_points[current_index:end_index, 0] = x
        grid_points[current_index:end_index, 1] = y
        grid_points[current_index:end_index, 2] = z_values

        # Update Index
        current_index = end_index
    if current_index < total_points:
        grid_points = grid_points[:current_index]

    return grid_points


def remove_overlapping_points(grid_points, sampling_points, tolerance):
    """Remove grid points coinciding with sampling points, utilising batch processing to enhance efficiency."""

    if len(sampling_points) == 0:
        return grid_points

    # Create a KDTree for efficient point matching
    sampling_tree = cKDTree(sampling_points)

    batch_size = 100000
    mask = np.ones(len(grid_points), dtype=bool)

    for i in tqdm(range(0, len(grid_points), batch_size), desc="Remove overlapping points"):
        batch = grid_points[i:i + batch_size]
        # Find grid points coinciding with sampling points
        dist_sampling, _ = sampling_tree.query(batch, k=1)
        # Update mask
        mask[i:i + batch_size] = dist_sampling > tolerance

    # Remove grid points coinciding with sampling points
    points_no_overlap = grid_points[mask]
    return points_no_overlap


def combine_points(grid_points, sampling_points, sampling_values):
    """Merge grid points and sampling points, and create an attribute array."""
    num_grid_points = grid_points.shape[0]
    num_sampling_points = sampling_points.shape[0]

    # Create an array of attributes for grid points
    property_grid = np.full((num_grid_points, sampling_values.shape[1]), np.nan)

    # Merge points and attributes
    all_points = np.vstack((grid_points, sampling_points)) if num_grid_points > 0 else sampling_points
    property_values = np.vstack((property_grid, sampling_values)) if num_grid_points > 0 else sampling_values
    return all_points, property_values


def scipy_delaunay_3d(points):
    """Performing 3D Delaunay triangulation using scipy.spatial"""
    start_time = time.time()

    #  Execute partitioning
    tetra = Delaunay(points)
    end_time = time.time()
    return tetra, end_time - start_time


def scipy_to_pyvista_mesh(points, tetra):
    """Convert SciPy's Delaunay results to a PyVista mesh"""
    # Create a tetrahedral mesh
    cells = np.hstack((np.full((len(tetra.simplices), 1), 4), tetra.simplices))
    mesh = pv.UnstructuredGrid(cells, np.ones(len(cells), dtype=np.int8) * 10, points)
    return mesh


def map_attributes_to_mesh(mesh, all_points, property_values):
    """Map attribute values to the subdivided mesh, 
    utilising batch processing to enhance efficiency when handling large datasets."""
    # KDTree for creating source points
    all_points_tree = cKDTree(all_points)
    batch_size = 100000
    indices_list = []
    distances_list = []

    for i in tqdm(range(0, len(mesh.points), batch_size), desc="Property mapping"):
        batch = mesh.points[i:i + batch_size]
        # Find the nearest point
        distances, indices = all_points_tree.query(batch, k=1)
        indices_list.append(indices)
        distances_list.append(distances)

    # Merged results
    indices = np.concatenate(indices_list)
    distances = np.concatenate(distances_list)

    # Verify the maximum distance to ensure the mapping is correct.
    max_distance = distances.max()
    if max_distance > 1e-6:
        print("Warning")

    # Retrieve property values
    node_attributes = property_values[indices]

    # Add attribute values to the point_data of the grid
    attribute_names = ['rock_unit', 'QJ', 'QX', 'level']
    for i, name in enumerate(attribute_names):
        mesh.point_data[name] = node_attributes[:, i]

    return node_attributes


def save_results(mesh, node_attributes, output_folder, missing_value):
    """Save the grid and node information to a file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    node_file = os.path.join(output_folder, 'combined_mesh.node')
    ele_file = os.path.join(output_folder, 'combined_mesh.ele')
    vtk_file = os.path.join(output_folder, 'combined_tetra_mesh.vtk')
    csv_file = os.path.join(output_folder, 'all_tetrahedron_nodes.csv')

    nodes = mesh.points
    # Acquire tetrahedral elements
    cells = mesh.cells.reshape(-1, 5)[:, 1:5]
    numerical_values = node_attributes[:, :-1].astype(float)  
    numerical_values = np.where(np.isnan(numerical_values), missing_value, numerical_values)
    node_attributes[:, :-1] = numerical_values

    # Save .node files, including the node's attribute values
    with open(node_file, 'w') as f:
        num_nodes = len(nodes)
        f.write(f"{num_nodes} 3 {node_attributes.shape[1]} 0\n")
        for i, (point, prop) in enumerate(zip(nodes, node_attributes)):
            f.write(f"{i + 1} {point[0]} {point[1]} {point[2]} {' '.join(map(str, prop))}\n")

    # Save the .ele file
    with open(ele_file, 'w') as f:
        num_elements = len(cells)
        f.write(f"{num_elements} 4 0\n")
        for i, cell in enumerate(cells):
            node_indices = cell + 1  
            f.write(f"{i + 1} {node_indices[0]} {node_indices[1]} {node_indices[2]} {node_indices[3]}\n")

    # Save nodes to a CSV file, including attribute values
    df_nodes = pd.DataFrame(nodes, columns=['X', 'Y', 'Z'])
    df_nodes[['rock_unit', 'QJ', 'QX', 'level']] = node_attributes
    df_nodes.to_csv(csv_file, index=False)
    mesh.save(vtk_file)


def visualize_mesh(mesh, node_attributes, missing_value, html_output_path=None):
    """Visualisation of grid and node properties"""
    export_image = False
    if html_output_path:
        try:
            import trame.widgets.vuetify
            export_html = True
        except ImportError:
            export_html = False
            export_image = True
            png_path = html_output_path.replace('.html', '.png')

    plotter = pv.Plotter(off_screen=export_image)

    plotter.add_mesh(
        mesh,
        style='points',
        point_size=5,
        render_points_as_spheres=True,
        scalars=node_attributes[:, 0], 
        nan_color='gray',
        cmap='viridis'
    )

    edges = mesh.extract_all_edges()
    plotter.add_mesh(edges, color='black', opacity=0.2, line_width=1)
    plotter.add_axes(
        interactive=True,
        color='white',
        line_width=2,
        labels_off=False
    )
    if html_output_path:
        if not os.path.exists(os.path.dirname(html_output_path)):
            os.makedirs(os.path.dirname(html_output_path))

        if export_html:
            try:
                plotter.export_html(html_output_path)
            except Exception as e:
                export_image = True
                png_path = html_output_path.replace('.html', '.png')

        if export_image:
            try:
                plotter.render()
                plotter.screenshot(png_path)
            except Exception as e:
                print(f"ERROR")

    if not export_image:
        plotter.show()


def reduce_points_for_delaunay(all_points, property_values, target_points=100000):
    """Reduce the number of points involved in the Delaunay triangulation while retaining key feature points."""
    if len(all_points) <= target_points:
        return all_points, property_values


    # 1. Retain all points with attribute values
    mask_has_attr = ~np.isnan(property_values).all(axis=1)
    points_with_attr = all_points[mask_has_attr]
    props_with_attr = property_values[mask_has_attr]

    # 2. Random sampling from the remaining points
    points_no_attr = all_points[~mask_has_attr]
    props_no_attr = property_values[~mask_has_attr]

    remaining_target = target_points - len(points_with_attr)
    if remaining_target <= 0:
        return all_points, property_values

    # Calculate the sampling rate
    sampling_rate = remaining_target / len(points_no_attr)

    np.random.seed(42) 
    random_indices = np.random.choice(
        len(points_no_attr),
        size=remaining_target,
        replace=False
    )

    sampled_points = points_no_attr[random_indices]
    sampled_props = props_no_attr[random_indices]

    # Merge points with attributes and sampled points
    reduced_points = np.vstack((points_with_attr, sampled_points))
    reduced_props = np.vstack((props_with_attr, sampled_props))
    return reduced_points, reduced_props


def compute_mesh_statistics(mesh):
    """Calculate grid statistics"""
    # Calculate per unit volume
    mesh = mesh.compute_cell_sizes()
    volumes = mesh.cell_data['Volume']

    # Statistical information
    total_volume = volumes.sum()
    average_volume = volumes.mean()
    min_volume = volumes.min()
    max_volume = volumes.max()

    # Output results
    print("\nMesh Statistics:")
    print(f"Total number of nodes: {mesh.n_points}")
    print(f"Number of tetrahedra: {mesh.n_cells}")
    print(f"Total volume: {total_volume:.2f}")
    print(f"Average tetrahedron volume: {average_volume:.2f}")
    print(f"Minimum tetrahedron volume: {min_volume:.2f}")
    print(f"Maximum tetrahedron volume: {max_volume:.2f}")

    # Create a volume distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(volumes, bins=50)
    plt.title('Tetrahedron Volume Distribution')
    plt.xlabel('Volume')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('F:/fzx/GCN/result/volume_distribution.png')

    return {
        'total_volume': total_volume,
        'average_volume': average_volume,
        'min_volume': min_volume,
        'max_volume': max_volume
    }


def main():
    # 参数设置
    output_folder = 'F:/fzx/GCN/Final_data/0605_50_50_50m'
    input_csv = 'F:/fzx/GCN/Data/allData_points.csv'
    dem_csv = 'F:/fzx/GCN/Data/DEM_dealed.csv'
    border_csv = 'F:/fzx/GCN/Data/border_dealed_new.csv'
    z_range = (-150, 1250)  # Z-range
    spacing_x = 50  # X-axis grid spacing
    spacing_y = 50  # Y-axis grid spacing
    spacing_z = 50  # Z-axis grid spacing
    tolerance = 1e-3  # Tolerance for coincident points
    missing_value = -9999  # Missing values
    target_points = 99999999  # Number of target points, controlling memory usage

    start_time_total = time.time()
    current_time=datetime.datetime.now()
    
    # Step 1: Load sampling points
    df = pd.read_csv(input_csv)
    sampling_points = df[['X', 'Y', 'Z']].values
    sampling_values = df[['rock_unit', 'QJ', 'QX', 'level']].values

    # Step 2: Load DEM data and boundaries
    dem_points = load_dem_data(dem_csv)
    boundary_points = load_boundary_points(border_csv)
    polygon = Polygon(boundary_points[:, :2])

    # Step 3: Filter sampling points
    sampling_points_filtered, mask = filter_sampling_points_by_polygon_and_z_range(
        sampling_points, polygon, z_range
    )
    sampling_values_filtered = sampling_values[mask]

    # Step 4: Generate grid points
    grid_points = generate_grid_points_below_dem(
        dem_points, z_range[0], spacing_x, spacing_y, spacing_z, polygon
    )

    # Step 5: Remove overlapping points
    points_no_overlap = remove_overlapping_points(
        grid_points, sampling_points_filtered, tolerance
    )

    # Step 6: Merge all points
    all_points, property_values = combine_points(
        points_no_overlap, sampling_points_filtered, sampling_values_filtered
    )

    # Step 7: If the number of points is excessive, perform point cloud simplification.
    reduced_points, reduced_props = reduce_points_for_delaunay(
        all_points, property_values, target_points
    )

    # Step 8: Performing Partitioning Using SciPy
    tetra, delaunay_duration = scipy_delaunay_3d(reduced_points)

    # Step 9: Convert to PyVista grid
    pv_mesh = scipy_to_pyvista_mesh(reduced_points, tetra)

    # Step 10: Map Properties
    node_attributes = map_attributes_to_mesh(pv_mesh, reduced_points, reduced_props)

    # Step 11: Calculate statistical information
    stats = compute_mesh_statistics(pv_mesh)

    # Step 12: Save the results
    save_results(pv_mesh, node_attributes, output_folder, missing_value)

    # Step 13: Visualisation
    visualize_mesh(
        pv_mesh,
        node_attributes,
        missing_value,
        html_output_path='F:/fzx/GCN/result/mesh_scene.html'
    )

    # Total output time
    end_time_total = time.time()


if __name__ == '__main__':
    main()
