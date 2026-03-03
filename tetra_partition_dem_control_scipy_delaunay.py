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
    """根据边界多边形和 Z 范围过滤采样点，保留位于多边形内部且 Z 值符合要求的点"""
    # 先过滤 Z 值（更高效的操作）
    mask_z = (sampling_points[:, 2] >= z_range[0]) & (sampling_points[:, 2] <= z_range[1])
    points_z_filtered = sampling_points[mask_z]

    if len(points_z_filtered) == 0:
        return points_z_filtered, mask_z

    # 再用多边形过滤（计算量更大的操作）
    xy_points = [Point(point[0], point[1]) for point in points_z_filtered]
    mask_polygon = np.array([polygon.contains(pt) for pt in xy_points], dtype=bool)

    # 更新总掩码
    final_mask = np.zeros_like(mask_z, dtype=bool)
    final_mask[mask_z] = mask_polygon

    return points_z_filtered[mask_polygon], final_mask


def load_boundary_points(boundary_csv):
    """加载剖分边界点的 X, Y, Z 数据"""
    df_boundary = pd.read_csv(boundary_csv)
    boundary_points = df_boundary[['X', 'Y', 'Z']].values
    return boundary_points


def load_dem_data(dem_csv):
    """加载 DEM 数据。假设 DEM 数据是一个 CSV 文件，包含 X, Y, Z 坐标。"""
    df_dem = pd.read_csv(dem_csv)
    dem_points = df_dem[['X', 'Y', 'Z']].values
    return dem_points


def is_point_in_polygon(point, polygon):
    """检查点是否在多边形内"""
    return polygon.contains(Point(point[0], point[1])) or polygon.intersects(Point(point[0], point[1]))


def is_points_batch_in_polygon(points_batch, polygon):
    """批量检查点是否在多边形内"""
    return np.array([polygon.contains(Point(pt[0], pt[1])) or
                     polygon.intersects(Point(pt[0], pt[1])) for pt in points_batch])


def generate_grid_points_below_dem(dem_points, z_min, spacing_x, spacing_y, spacing_z, polygon):
    """优化的网格点生成算法，使用批处理和预分配内存"""
    print("开始生成网格点...")

    # 获取 DEM 的 XY 范围（通过多边形边界的最大最小值）
    x_min, x_max = polygon.bounds[0], polygon.bounds[2]
    y_min, y_max = polygon.bounds[1], polygon.bounds[3]

    # 使用不同的间距预先生成 X、Y 范围
    x_range = np.arange(x_min, x_max + spacing_x, spacing_x)
    y_range = np.arange(y_min, y_max + spacing_y, spacing_y)

    # 创建 XY 网格
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    xy_points = np.vstack([X.ravel(), Y.ravel()]).T

    # 批量检查点是否在多边形内
    print("检查点是否在多边形边界内...")
    batch_size = 10000  # 设置批量大小以平衡内存使用和速度
    in_polygon_mask = np.zeros(len(xy_points), dtype=bool)

    for i in tqdm(range(0, len(xy_points), batch_size), desc="多边形检查"):
        batch = xy_points[i:i + batch_size]
        in_polygon_mask[i:i + batch_size] = is_points_batch_in_polygon(batch, polygon)

    # 只保留多边形内的点
    xy_points_filtered = xy_points[in_polygon_mask]

    print(f"多边形内的 XY 点数量: {len(xy_points_filtered)}")

    # 构建 DEM 的 KDTree 用于快速查找
    dem_tree = cKDTree(dem_points[:, :2])

    # 查询 DEM 上的 Z 值
    print("查询 DEM 上的 Z 值...")
    dist, indices = dem_tree.query(xy_points_filtered, k=1)
    dem_z_values = dem_points[indices, 2]

    # 计算每个点要生成的 Z 值数量
    z_points_per_xy = np.ceil((dem_z_values - z_min) / spacing_z).astype(int) + 1
    total_points = np.sum(z_points_per_xy)

    print(f"预计生成网格点总数: {total_points}")

    # 预分配最终点数组
    grid_points = np.zeros((total_points, 3))

    # 填充点数组
    print("生成 Z 值和最终网格点...")
    current_index = 0
    for i, (x, y, top_z, num_z) in enumerate(zip(xy_points_filtered[:, 0],
                                                 xy_points_filtered[:, 1],
                                                 dem_z_values,
                                                 z_points_per_xy)):
        # 生成这个 xy 位置的所有 z 值
        z_values = np.linspace(top_z, z_min, num_z)

        # 计算这些点在最终数组中的索引
        end_index = current_index + len(z_values)

        # 填充 XYZ 坐标
        grid_points[current_index:end_index, 0] = x
        grid_points[current_index:end_index, 1] = y
        grid_points[current_index:end_index, 2] = z_values

        # 更新索引
        current_index = end_index

    # 如果预估不准确，可能需要裁剪数组
    if current_index < total_points:
        grid_points = grid_points[:current_index]

    print(f"实际生成的网格点数量: {len(grid_points)}")
    return grid_points


def remove_overlapping_points(grid_points, sampling_points, tolerance):
    """删除与采样点重合的网格点，使用批处理提高效率。"""
    print("移除重叠点...")

    if len(sampling_points) == 0:
        return grid_points

    # 创建 KDTree 进行高效的点匹配
    sampling_tree = cKDTree(sampling_points)

    # 批量处理以节省内存
    batch_size = 100000
    mask = np.ones(len(grid_points), dtype=bool)

    for i in tqdm(range(0, len(grid_points), batch_size), desc="移除重叠点"):
        batch = grid_points[i:i + batch_size]
        # 查找与采样点重合的网格点
        dist_sampling, _ = sampling_tree.query(batch, k=1)
        # 更新掩码
        mask[i:i + batch_size] = dist_sampling > tolerance

    # 删除与采样点重合的网格点
    points_no_overlap = grid_points[mask]
    print(f"移除重叠点后的点数量: {len(points_no_overlap)}")
    return points_no_overlap


def combine_points(grid_points, sampling_points, sampling_values):
    """合并网格点和采样点，并创建属性数组。"""
    num_grid_points = grid_points.shape[0]
    num_sampling_points = sampling_points.shape[0]

    # 创建网格点的属性数组（全部为NaN）
    property_grid = np.full((num_grid_points, sampling_values.shape[1]), np.nan)

    # 合并点和属性
    all_points = np.vstack((grid_points, sampling_points)) if num_grid_points > 0 else sampling_points
    property_values = np.vstack((property_grid, sampling_values)) if num_grid_points > 0 else sampling_values

    print(f"合并后的总点数: {len(all_points)}")
    return all_points, property_values


def scipy_delaunay_3d(points):
    """使用scipy.spatial进行3D Delaunay剖分"""
    print(f"开始3D Delaunay剖分，总计 {len(points)} 个点...")
    start_time = time.time()

    # 执行剖分
    tetra = Delaunay(points)

    end_time = time.time()
    print(f"剖分完成，用时: {end_time - start_time:.2f} 秒")
    print(f"生成了 {len(tetra.simplices)} 个四面体单元")

    return tetra, end_time - start_time


def scipy_to_pyvista_mesh(points, tetra):
    """将SciPy的Delaunay结果转换为PyVista网格"""
    # 创建四面体网格
    cells = np.hstack((np.full((len(tetra.simplices), 1), 4), tetra.simplices))
    mesh = pv.UnstructuredGrid(cells, np.ones(len(cells), dtype=np.int8) * 10, points)
    return mesh


def map_attributes_to_mesh(mesh, all_points, property_values):
    """将属性值映射到剖分后的网格，使用批处理提高大型数据集的处理效率。"""
    print("映射属性值到网格节点...")

    # 创建源点的KDTree
    all_points_tree = cKDTree(all_points)

    # 批量处理查询
    batch_size = 100000
    indices_list = []
    distances_list = []

    for i in tqdm(range(0, len(mesh.points), batch_size), desc="属性映射"):
        batch = mesh.points[i:i + batch_size]
        # 查找最近点
        distances, indices = all_points_tree.query(batch, k=1)
        indices_list.append(indices)
        distances_list.append(distances)

    # 合并结果
    indices = np.concatenate(indices_list)
    distances = np.concatenate(distances_list)

    # 检查最大距离，确保映射正确
    max_distance = distances.max()
    print(f"最大映射距离: {max_distance}")
    if max_distance > 1e-6:
        print("警告：一些剖分后的节点无法准确映射到原始点。")

    # 获取属性值
    node_attributes = property_values[indices]

    # 将属性值添加到网格的 point_data 中
    attribute_names = ['rock_unit', 'QJ', 'QX', 'level']
    for i, name in enumerate(attribute_names):
        mesh.point_data[name] = node_attributes[:, i]

    return node_attributes


def save_results(mesh, node_attributes, output_folder, missing_value):
    """保存网格和节点信息到文件。"""
    print("保存结果...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    node_file = os.path.join(output_folder, 'combined_mesh.node')
    ele_file = os.path.join(output_folder, 'combined_mesh.ele')
    vtk_file = os.path.join(output_folder, 'combined_tetra_mesh.vtk')
    csv_file = os.path.join(output_folder, 'all_tetrahedron_nodes.csv')

    nodes = mesh.points
    # 获取四面体单元
    cells = mesh.cells.reshape(-1, 5)[:, 1:5]

    # 单独处理数值部分
    numerical_values = node_attributes[:, :-1].astype(float)  # 假设最后一列是字符串或其他类型
    numerical_values = np.where(np.isnan(numerical_values), missing_value, numerical_values)
    # 合并数值部分和字符串部分（如果有）
    node_attributes[:, :-1] = numerical_values

    # 保存 .node 文件，包括节点的属性值
    with open(node_file, 'w') as f:
        num_nodes = len(nodes)
        f.write(f"{num_nodes} 3 {node_attributes.shape[1]} 0\n")
        for i, (point, prop) in enumerate(zip(nodes, node_attributes)):
            f.write(f"{i + 1} {point[0]} {point[1]} {point[2]} {' '.join(map(str, prop))}\n")

    # 保存 .ele 文件
    with open(ele_file, 'w') as f:
        num_elements = len(cells)
        f.write(f"{num_elements} 4 0\n")
        for i, cell in enumerate(cells):
            node_indices = cell + 1  # 节点编号从 1 开始
            f.write(f"{i + 1} {node_indices[0]} {node_indices[1]} {node_indices[2]} {node_indices[3]}\n")

    # 保存节点到 CSV 文件，包括属性值
    df_nodes = pd.DataFrame(nodes, columns=['X', 'Y', 'Z'])
    df_nodes[['rock_unit', 'QJ', 'QX', 'level']] = node_attributes
    df_nodes.to_csv(csv_file, index=False)

    # 保存网格为 VTK 文件，包括属性值
    mesh.save(vtk_file)
    print(f"结果已保存到文件夹: {output_folder}")


def visualize_mesh(mesh, node_attributes, missing_value, html_output_path=None):
    """可视化网格和节点属性"""
    print("可视化网格...")

    # 检查是否需要导出HTML或截图
    export_image = False
    if html_output_path:
        try:
            import trame.widgets.vuetify
            export_html = True
        except ImportError:
            print("警告: 无法导出HTML文件，因为缺少trame库。")
            print("请使用以下命令安装trame: pip install trame[all] --upgrade")
            export_html = False
            export_image = True
            png_path = html_output_path.replace('.html', '.png')

    # 如果需要截图，则设置off_screen=True
    plotter = pv.Plotter(off_screen=export_image)

    # 可视化所有节点
    plotter.add_mesh(
        mesh,
        style='points',
        point_size=5,
        render_points_as_spheres=True,
        scalars=node_attributes[:, 0],  # 默认使用第一个属性进行着色
        nan_color='gray',
        cmap='viridis'
    )

    # 提取所有边
    edges = mesh.extract_all_edges()

    # 可视化边，设置透明度
    plotter.add_mesh(edges, color='black', opacity=0.2, line_width=1)

    # 添加三维坐标轴
    plotter.add_axes(
        interactive=True,
        color='white',
        line_width=2,
        labels_off=False
    )

    # 导出文件
    if html_output_path:
        if not os.path.exists(os.path.dirname(html_output_path)):
            os.makedirs(os.path.dirname(html_output_path))

        if export_html:
            try:
                plotter.export_html(html_output_path)
                print(f"已成功导出HTML到: {html_output_path}")
            except Exception as e:
                print(f"HTML导出失败: {str(e)}")
                export_image = True
                png_path = html_output_path.replace('.html', '.png')

        if export_image:
            try:
                # 确保渲染器已初始化
                plotter.render()
                # 保存截图
                plotter.screenshot(png_path)
                print(f"已保存截图到: {png_path}")
            except Exception as e:
                print(f"保存截图失败: {str(e)}")

    # 显示绘图 (如果不是离屏渲染)
    if not export_image:
        plotter.show()


def reduce_points_for_delaunay(all_points, property_values, target_points=100000):
    """减少参与Delaunay剖分的点数量，保留重要特征点"""
    if len(all_points) <= target_points:
        return all_points, property_values

    print(f"点数量({len(all_points)})超过目标值({target_points})，执行点云简化...")

    # 1. 保留所有有属性值的点（通常是采样点）
    mask_has_attr = ~np.isnan(property_values).all(axis=1)
    points_with_attr = all_points[mask_has_attr]
    props_with_attr = property_values[mask_has_attr]

    # 2. 从剩余点中随机抽样
    points_no_attr = all_points[~mask_has_attr]
    props_no_attr = property_values[~mask_has_attr]

    remaining_target = target_points - len(points_with_attr)
    if remaining_target <= 0:
        print("警告：采样点数量已超过目标点数，不进行简化")
        return all_points, property_values

    # 计算抽样率
    sampling_rate = remaining_target / len(points_no_attr)
    print(f"采样率: {sampling_rate:.4f}")

    # 随机抽样
    np.random.seed(42)  # 设置随机种子以确保结果可重复
    random_indices = np.random.choice(
        len(points_no_attr),
        size=remaining_target,
        replace=False
    )

    sampled_points = points_no_attr[random_indices]
    sampled_props = props_no_attr[random_indices]

    # 合并有属性的点和抽样的点
    reduced_points = np.vstack((points_with_attr, sampled_points))
    reduced_props = np.vstack((props_with_attr, sampled_props))

    print(f"简化后的点数量: {len(reduced_points)}")
    return reduced_points, reduced_props


def compute_mesh_statistics(mesh):
    """计算网格统计信息"""
    # 计算单元体积
    mesh = mesh.compute_cell_sizes()
    volumes = mesh.cell_data['Volume']

    # 统计信息
    total_volume = volumes.sum()
    average_volume = volumes.mean()
    min_volume = volumes.min()
    max_volume = volumes.max()

    # 输出结果
    print("\nMesh Statistics:")
    print(f"Total number of nodes: {mesh.n_points}")
    print(f"Number of tetrahedra: {mesh.n_cells}")
    print(f"Total volume: {total_volume:.2f}")
    print(f"Average tetrahedron volume: {average_volume:.2f}")
    print(f"Minimum tetrahedron volume: {min_volume:.2f}")
    print(f"Maximum tetrahedron volume: {max_volume:.2f}")

    # 创建体积分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(volumes, bins=50)
    plt.title('Tetrahedron Volume Distribution')
    plt.xlabel('Volume')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图表
    plt.savefig('F:/fzx/GCN/result/volume_distribution.png')
    print("体积分布直方图已保存到 F:/fzx/GCN/result/volume_distribution.png")

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
    z_range = (-150, 1250)  # Z范围
    spacing_x = 50  # X方向网格间隔
    spacing_y = 50  # Y方向网格间隔
    spacing_z = 50  # Z方向网格间隔
    tolerance = 1e-3  # 重合点的容忍度
    missing_value = -9999  # 缺失值
    target_points = 99999999  # 目标点数量，控制内存使用

    start_time_total = time.time()
    current_time=datetime.datetime.now()
    # 步骤1：加载采样点
    print("开始时间：",current_time)
    print("1. 加载采样点数据...")
    df = pd.read_csv(input_csv)
    sampling_points = df[['X', 'Y', 'Z']].values
    sampling_values = df[['rock_unit', 'QJ', 'QX', 'level']].values

    # 步骤2：加载DEM数据和边界
    print("2. 加载DEM和边界数据...")
    dem_points = load_dem_data(dem_csv)
    boundary_points = load_boundary_points(border_csv)
    polygon = Polygon(boundary_points[:, :2])

    # 步骤3：过滤采样点
    print("3. 过滤采样点...")
    sampling_points_filtered, mask = filter_sampling_points_by_polygon_and_z_range(
        sampling_points, polygon, z_range
    )
    sampling_values_filtered = sampling_values[mask]

    # 步骤4：生成网格点（优化版本）
    print("4. 生成网格点...")
    grid_points = generate_grid_points_below_dem(
        dem_points, z_range[0], spacing_x, spacing_y, spacing_z, polygon
    )

    # 步骤5：去除重合点
    print("5. 移除重合点...")
    points_no_overlap = remove_overlapping_points(
        grid_points, sampling_points_filtered, tolerance
    )

    # 步骤6：合并所有点
    print("6. 合并点集...")
    all_points, property_values = combine_points(
        points_no_overlap, sampling_points_filtered, sampling_values_filtered
    )

    # 步骤7：如果点数过多，执行点云简化
    print("7. 检查点数并简化（如需要）...")
    reduced_points, reduced_props = reduce_points_for_delaunay(
        all_points, property_values, target_points
    )

    # 步骤8：使用SciPy进行剖分
    print("8. 执行Delaunay剖分...")
    tetra, delaunay_duration = scipy_delaunay_3d(reduced_points)

    # 步骤9：转换为PyVista网格
    print("9. 转换为PyVista网格...")
    pv_mesh = scipy_to_pyvista_mesh(reduced_points, tetra)

    # 步骤10：映射属性
    print("10. 映射属性值...")
    node_attributes = map_attributes_to_mesh(pv_mesh, reduced_points, reduced_props)

    # 步骤11：计算统计信息
    print("11. 计算网格统计信息...")
    stats = compute_mesh_statistics(pv_mesh)

    # 步骤12：保存结果
    print("12. 保存结果...")
    save_results(pv_mesh, node_attributes, output_folder, missing_value)

    # 步骤13：可视化
    print("13. 可视化网格...")
    visualize_mesh(
        pv_mesh,
        node_attributes,
        missing_value,
        html_output_path='F:/fzx/GCN/result/mesh_scene.html'
    )

    # 输出总时间
    end_time_total = time.time()
    print(f"\n总处理时间: {end_time_total - start_time_total:.2f} 秒")


if __name__ == '__main__':
    main()