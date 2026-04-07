import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize, ball, erosion
from scipy import ndimage as ndi
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import SimpleITK as sitk
import napari
from .spline import catmull_rom_spline

# -------------------------
#  配置参数
# -------------------------

control_point_num = 10  # 选取控制点的数量
cube_size = 24  # 3D立方体的直径

# -------------------------

# ---------- 辅助函数 ----------
def image_resample(img, new_spacing, is_label=False):
    """重采样图像到新的体素间距"""
    spacing = np.array(img.GetSpacing())
    size = np.array(img.GetSize())
    new_size = np.round(size * (spacing / new_spacing)).astype(int).tolist()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    new_img = resample.Execute(img)
    return new_img


def load_label(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img).transpose(2, 1, 0)  # 转为 x,y,z 顺序
    return data.astype(np.uint8), img.GetSpacing(), img.GetOrigin(), img.GetDirection()


def save_nifti(arr, affine, header, out_path):
    nii = nib.Nifti1Image(arr.astype(np.uint8), affine, header)
    nib.save(nii, out_path)


def transform_coordinates(coords, spacing, origin, direction):
    """将坐标从图像空间转换到世界空间"""
    coords = np.array(coords)
    coords = coords * spacing + origin
    coords = np.dot(coords, direction)
    return coords


# ---------- 3D 骨架化 ----------
def skeletonize_volume(binary_vol):
    """
    输入:binary_vol (3D ndarray, 0/1)
    输出:skeleton (3D ndarray, 0/1)
    """
    skel = skeletonize(
        binary_vol, method="lee"
    )  # scikit-image 的 3D 骨架化 (lee方法为3D的默认算法)
    return skel.astype(np.uint8)


def erosion_volume(binary_vol, radius=1):
    """
    使用球形结构元素对二值标注进行腐蚀
    """
    selem = ball(radius)
    eroded_vol = erosion(binary_vol, selem)
    return eroded_vol


def view_skeleton_napari(skel):
    """使用 napari 可视化 3D 骨架"""
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_labels(skel, name="Skeleton")
    napari.run()


# ---------- 从骨架提取图（节点=端点/交点，边=像素链） ----------
def dfs(graph, start_node, cur_node, node_list, skeleton, visited, direction, coord2idx, path):
    visited[cur_node] = True
    path = path + [cur_node]
    for dx, dy, dz in direction:
        neighbor = (cur_node[0] + dx, cur_node[1] + dy, cur_node[2] + dz)
        if (
            0 <= neighbor[0] < skeleton.shape[0]
            and 0 <= neighbor[1] < skeleton.shape[1]
            and 0 <= neighbor[2] < skeleton.shape[2]
            and skeleton[neighbor] == 1
            and not visited[neighbor]
        ):
            # 如果下一个邻居是节点，则记录一条边
            if neighbor in node_list:
                path.append(neighbor)
                graph.add_edge(
                    coord2idx[start_node], coord2idx[neighbor], 
                    pixels=path.copy(), length=len(path)
                )
                dfs(
                    graph,
                    neighbor,
                    neighbor,
                    node_list,
                    skeleton,
                    visited,
                    direction,
                    coord2idx,
                    path=[],
                )
            else:
                dfs(
                    graph,
                    start_node,
                    neighbor,
                    node_list,
                    skeleton,
                    visited,
                    direction,
                    coord2idx,
                    path,
                )


def input_edges(skeleton, node_coord, coord2idx, G):
    visited = np.zeros_like(skeleton, dtype=bool)
    direction = [
        (i, j, k)
        for i in [-1, 0, 1]
        for j in [-1, 0, 1]
        for k in [-1, 0, 1]
        if not (i == 0 and j == 0 and k == 0)
    ]
    for node in node_coord:
        dfs(G, node, node, node_coord, skeleton, visited, direction, coord2idx, path=[])


def skeleton_to_graph(skeleton):
    """
    输入: 3D binary skeleton
    输出: (nodes, edges)
      nodes: list of node coordinate (z,y,x)
      edges: list of (nodeA, nodeB, path_voxels)
    作用: 把骨架像素链分解为“节点”和“边”
    """
    sk = skeleton.copy().astype(np.uint8)
    G = nx.Graph()

    # 26 邻域卷积核
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neigh_count = ndi.convolve(sk, kernel, mode="constant", cval=0)

    # 节点：端点(degree==1)和交点(degree>=3)
    node_mask = (sk == 1) & (neigh_count == 1) | (sk == 1) & (neigh_count >= 3)
    node_coord = [tuple(coord) for coord in np.argwhere(node_mask)]
    coord2idx = {coord: idx for idx, coord in enumerate(node_coord)}
    for idx, coord in enumerate(node_coord):
        G.add_node(idx, pos=coord)

    # 边：遍历所有节点，沿骨架像素链走访，直到遇到另一个节点
    input_edges(sk, node_coord, coord2idx, G)

    return G


def merge_nodes(graph):
    visited = set()  # 记录已经处理过的节点

    for node in list(graph.nodes()):
        # 跳过已经访问的节点或度不是1或>2的节点
        if node in visited or graph.degree(node) == 2:
            continue

        # 链条追踪从链头开始
        head = node
        neighbors = list(graph.neighbors(head))

        for neighbor in neighbors:
            if neighbor in visited:
                continue

            # 初始化链条
            chain_nodes = [head]
            prev = head
            current = neighbor

            while graph.degree(current) == 2:
                chain_nodes.append(current)
                visited.add(current)

                # 找到下一个节点
                next_nodes = [n for n in graph.neighbors(current) if n != prev]
                if not next_nodes:
                    break  # 到达链尾
                prev, current = current, next_nodes[0]

            # current 是链尾节点（度不为2），也加入链条
            tail = current
            chain_nodes.append(tail)
            visited.add(tail)

            # 如果链条长度 <=2，说明没有可合并节点
            if len(chain_nodes) <= 2:
                continue

            # 合并路径
            merged_path = np.array(
                graph.edges[chain_nodes[0], chain_nodes[1]]["pixels"]
            )
            for i in range(1, len(chain_nodes) - 1):
                path = np.array(
                    graph.edges[chain_nodes[i], chain_nodes[i + 1]]["pixels"]
                )

                # 方向对齐
                if not np.array_equal(merged_path[-1], path[0]):
                    if np.array_equal(merged_path[-1], path[-1]):
                        path = path[::-1]
                    elif np.array_equal(merged_path[0], path[0]):
                        merged_path = merged_path[::-1]
                    elif np.array_equal(merged_path[0], path[-1]):
                        merged_path = merged_path[::-1]
                        path = path[::-1]

                merged_path = np.vstack((merged_path, path[1:]))

            # 添加新边
            graph.add_edge(
                chain_nodes[0],
                chain_nodes[-1],
                pixels=merged_path,
                length=len(merged_path),
            )

            # 删除中间节点（不包括链头和链尾）
            to_remove = chain_nodes[1:-1]
            graph.remove_nodes_from(to_remove)
            visited.update(to_remove)


def delete_short_edges(graph, min_length=5):
    edges_to_remove = []
    for u, v, data in graph.edges(data=True):
        if data["length"] < min_length and (
            graph.degree(u) == 1 or graph.degree(v) == 1
        ):
            edges_to_remove.append((u, v))
    graph.remove_edges_from(edges_to_remove)
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)

    mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
    graph = nx.relabel_nodes(graph, mapping)

    return graph


# def select_control_points(graph, num_points=control_point_num):
#     """
#     从边中均匀选取控制点
#     """
#     for u, v in graph.edges():
#         length = graph.edges[u, v]["length"]
#         path = np.array(graph.edges[u, v]["pixels"])
#         if length <= num_points:
#             continue
#         indices = np.linspace(0, length - 1, num_points).astype(int)
#         control_points = path[indices]
#         graph.edges[u, v]["control_points"] = control_points

def select_control_points(graph, step=10):
    """
    每隔固定步长选取控制点，保证稀疏且均匀
    """
    for u, v in graph.edges():
        length = graph.edges[u, v]["length"]
        path = np.array(graph.edges[u, v]["pixels"])
        indices = np.arange(0, length, step)
        if indices[-1] != length - 1:
            indices = np.append(indices, length - 1)  # 确保包含末尾点
        control_points = path[indices]
        graph.edges[u, v]["control_points"] = control_points

def smooth_edge(graph):
    """
    使用 Catmull-Rom 样条平滑边的像素链
    """
    for u, v in graph.edges():
        if "control_points" not in graph.edges[u, v]:
            continue
        control_points = graph.edges[u, v]["control_points"]
        smoothed_path = catmull_rom_spline(control_points, num_points=50)
        graph.edges[u, v]["centerline"] = smoothed_path 


# ---------- 特征提取 ----------
def convert_to_SCT(points, eps=1e-8):
    """
    将像素坐标转换为SCT坐标系下的坐标
    """
    # 1. 计算r, cos(theta), sin(phi)
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
    r_safe = np.maximum(r, eps)
    cos_theta = np.where(r != 0, points[:, 2] / r_safe, 1)
    sin_theta = np.where(
        r != 0, np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) / r_safe, 0
    )
    sin_theta_safe = np.maximum(sin_theta, eps)
    sin_phi = np.where(sin_theta != 0, points[:, 0] / (r_safe * sin_theta_safe), 0)
    cos_phi = np.where(sin_theta != 0, points[:, 1] / (r_safe * sin_theta_safe), 1)

    # 2. 转换为M矩阵
    M = np.stack(
        [
            np.stack([sin_theta, sin_phi], axis=-1),
            np.stack([cos_theta, cos_phi], axis=-1),
        ],
        axis=1,
    )
    return M

def safe_normalize(v, eps=1e-8):
    norm = np.linalg.norm(v)
    if norm < eps:
        return None
    return v / norm

def normalize_3d_positions(points):
    """
    对3D坐标进行归一化处理
    """
    if points.shape[0] < 2:
        print("There exists an edge that contains two vertices.")
    
    # 1. 去平移
    points_relative = points - points[0]

    # 2. 构造 z 轴（首段方向）
    z_axis = safe_normalize(points[1] - points[0])
    if z_axis is None:
        raise ValueError("The first segment has a length of 0")

    # 3. 构造 y 轴（首点 -> 末点）
    temp_y = safe_normalize(points[-1] - points[0])
    if temp_y is None:
        # 所有点重合，退化情况
        return np.zeros_like(points), z_axis, np.zeros(3)

    # 4. 构造 x 轴（处理共线）
    x_axis = safe_normalize(np.cross(temp_y, z_axis))
    if x_axis is None:
        # temp_y 与 z_axis 共线，换参考向量
        if abs(z_axis[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        x_axis = safe_normalize(np.cross(ref, z_axis))

    # 5. 构造 y 轴（保证正交）
    y_axis = np.cross(z_axis, x_axis)

    # 6. 旋转
    R = np.vstack([x_axis, y_axis, z_axis]).T
    points_rot = points_relative @ R

    # 7. 去尺度
    seg_lengths = np.linalg.norm(np.diff(points_rot, axis=0), axis=1)
    total_length = np.sum(seg_lengths)

    if total_length < 1e-8:
        points_norm = np.zeros_like(points_rot)
    else:
        points_norm = points_rot / total_length

    return points_norm, z_axis, y_axis

def extract_position_features(graph):
    """
    根据录入的边的像素信息，提取边的长度特征
    选取特征：
        (1) 第一个点、中心点和最后一个点的S2投影和归一化3D位置。
        (2) 第一个点和最后一个点之间的方向向量 和 起始点处的切线方向。
    """
    for u, v, data in graph.edges(data=True):
        if "centerline" in data:
            path = np.array(data["centerline"])
        else:
            path = np.array(data["pixels"])

        # 进行坐标的归一化与S2投影
        path_norm, z_axis, y_axis = normalize_3d_positions(path)
        M = convert_to_SCT(path_norm)

        # 记录起始点、中点与末尾点
        node_3d = np.array([path_norm[0], path_norm[len(path_norm) // 2], path_norm[-1]])
        node_SCT = np.array([M[0], M[len(M) // 2], M[-1]])

        graph.edges[u, v]["node_3d"] = node_3d.reshape(1, -1)
        graph.edges[u, v]["node_SCT"] = node_SCT.reshape(1, -1)
        graph.edges[u, v]["z_axis"] = z_axis.reshape(1, -1)
        graph.edges[u, v]["y_axis"] = y_axis.reshape(1, -1)

def extract_img_features(graph, img_array, cube_size=cube_size):
    """
    在图的每条边的 control_points 位置提取 3D 影像 patch
    """
    
    # 获取每条边的图像特征序列
    half = cube_size // 2
    W, D, H = img_array.shape  # correspond x,y,z order
    for u, v, data in graph.edges(data=True):

        patches = []
        control_points = data.get("control_points", [])
        for pt in control_points:
            x, y, z = map(int, pt)

            # 计算patch边界
            x1, x2 = x - half, x + half
            y1, y2 = y - half, y + half
            z1, z2 = z - half, z + half

            # 创建空patch
            patch = np.zeros((cube_size, cube_size, cube_size), dtype=img_array.dtype)

            # 计算有效范围（避免越界）
            xs1, xs2 = max(0, x1), min(W, x2)
            ys1, ys2 = max(0, y1), min(D, y2)
            zs1, zs2 = max(0, z1), min(H, z2)

            # 对应到patch中的位置
            px1, px2 = xs1 - x1, xs2 - x1
            py1, py2 = ys1 - y1, ys2 - y1
            pz1, pz2 = zs1 - z1, zs2 - z1

            # 复制数据
            patch[px1:px2, py1:py2, pz1:pz2] = img_array[xs1:xs2, ys1:ys2, zs1:zs2]
            patches.append(patch)

        graph.edges[(u, v)]["image"] = patches

def assign_edge_labels(graph, label_volume):
    """
    为每条边分配标签（沿边像素链在 label_volume 上做多数投票）
    """
    x_max, y_max, z_max = label_volume.shape
    for u, v, data in graph.edges(data=True):
        path = np.array(data["pixels"])
        coords = np.rint(path).astype(np.int64)
        coords[:, 0] = np.clip(coords[:, 0], 0, x_max - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, y_max - 1)
        coords[:, 2] = np.clip(coords[:, 2], 0, z_max - 1)

        voxel_labels = label_volume[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.int64)
        voxel_labels = voxel_labels[voxel_labels > 0]

        if voxel_labels.size == 0:
            graph.edges[u, v]["label"] = -1
            continue

        majority_label = np.bincount(voxel_labels).argmax()
        graph.edges[u, v]["label"] = int(majority_label - 1)

# ---------- 可视化 ----------
def plot_graph_3d(graph):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 打印平滑化结果
    for u, v, data in graph.edges(data=True):
        if "centerline" in data:
            path = np.array(data["centerline"])
        else:
            path = np.array(data["pixels"])
        ax.plot(
            path[:, 0], path[:, 1], path[:, 2], "r-", linewidth=0.5
        )
        path = np.array(data["pixels"])
        ax.plot(
            path[:, 0], path[:, 1], path[:, 2], "b-", linewidth=0.5
        )
    
    #打印节点
    for node, attr in graph.nodes(data=True):
        coord = attr["pos"]  # 坐标
        deg = graph.degree(node)

        if deg == 1:
            ax.scatter(coord[0], coord[1], coord[2], c="g", s=50, marker="^")  # 端点
        elif deg >= 3:
            ax.scatter(coord[0], coord[1], coord[2], c="b", s=50, marker="o")  # 交点
    plt.show()

def plot_graph_3d_with_label(graph, label_array):
    """
    使用传入的 label_array 给图的边上色绘制 3D 图
    
    参数:
        graph: NetworkX 图对象，要求每个节点有 "pos" 属性，每条边有 "pixels" 或 "centerline"
        label_array: np.array 或 list, 长度 = 图中边数，对应每条边的类别
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    color_map = {0: "r", 1: "g", 2: "b", 3: "c", 4: "m"}  # 可以根据类别自定义颜色

    # edges 按顺序列表
    edges = list(graph.edges())
    
    if len(edges) != len(label_array):
        raise ValueError(f"label_array 长度 {len(label_array)} 与图边数 {len(edges)} 不匹配")

    # 给每条边加 label，然后绘制
    for i, (u, v) in enumerate(edges):
        data = graph.edges[u, v]
        data['label'] = int(label_array[i])  # 把 label_array 映射到边上
        
        # 绘制边
        path = np.array(data.get("pixels"))
        color = color_map.get(data['label'], "k")  # 默认黑色
        ax.plot(path[:,0], path[:,1], path[:,2], color=color, linewidth=0.5)
    
    # 绘制节点
    for node, attr in graph.nodes(data=True):
        coord = attr["pos"]
        deg = graph.degree(node)
        if deg == 1:
            ax.scatter(coord[0], coord[1], coord[2], c="g", s=50, marker="^")  # 端点
        elif deg >= 3:
            ax.scatter(coord[0], coord[1], coord[2], c="b", s=50, marker="o")  # 交点

    plt.show()

def dump_graph(graph):
    print("Graph info:")
    print(f"node nums:{graph.number_of_nodes()}")
    print(f"edge nums:{graph.number_of_edges()}")
    print("Nodes with attributes:")
    for node, attr in graph.nodes(data=True):
        print(node, attr)

    print("Edges with attributes:")
    for u, v, attr in graph.edges(data=True):
        print(f"node:{(u, v)}, origin_length:{attr['length']}\n\
              control_points_num:{control_point_num}\
            ")
        print(attr["length"])
        print(attr["control_points"].shape)
        print(attr["centerline"].shape)
        print(attr["node_3d"])
        print(attr["node_SCT"])
        print(attr["z_axis"])
        print(attr["y_axis"])

# ---------- 主流程 ----------
def data_process(data_path, img_path):
    data = sitk.ReadImage(data_path)  # 512*512*275
    img = sitk.ReadImage(img_path)  # 512*512*275

    # 对影像、冠脉模型、标注进行重采样
    new_spacing = (0.5, 0.5, 0.5)  # 目标体素间距(mm)
    img_resampled = image_resample(img, new_spacing, is_label=False)
    data_resampled = image_resample(data, new_spacing, is_label=False)
    label_resampled = image_resample(data, new_spacing, is_label=True)

    # 转为 numpy 数组 （保持 x,y,z 顺序）
    img_array = sitk.GetArrayFromImage(img_resampled).transpose(2, 1, 0)
    data_array = sitk.GetArrayFromImage(data_resampled).transpose(2, 1, 0)
    label_array = sitk.GetArrayFromImage(label_resampled).transpose(2, 1, 0)

    # 将冠脉整体进行二值化与腐蚀
    bin_vol = (data_array > 0).astype(np.uint8)
    erosion_vol = erosion_volume(bin_vol, radius=1)

    # 提取骨架
    skel = skeletonize_volume(erosion_vol)

    # 构建图结构
    graph = skeleton_to_graph(skel)
    merge_nodes(graph)
    graph = delete_short_edges(graph, min_length=5)

    # 平滑中心线
    select_control_points(graph, step=10)
    smooth_edge(graph)

    # 计算位置域特征
    extract_position_features(graph)

    # 计算图像域特征
    extract_img_features(graph, img_array)

    # 分配标签
    assign_edge_labels(graph, label_array)

    return graph
