import gmsh
import json
import numpy as np

from typing import Tuple, Dict
from numpy.typing import NDArray


def load_mesh_nodes():
    nodeTags, coord, _ = gmsh.model.mesh.getNodes()

    # Convert to (num_nodes, 2)
    coords = np.array(coord).reshape(-1, 3)[:, :2]

    # Map nodeTag -> index
    tag_to_index = {tag: i for i, tag in enumerate(nodeTags)}

    return nodeTags, coords, tag_to_index


def load_line_elements(tag_to_index):
    _, _, elemNodeTags = gmsh.model.mesh.getElements(dim=1)

    edges = []

    for elem_group in elemNodeTags:
        elem_group = np.array(elem_group).reshape(-1, 2)
        for n1, n2 in elem_group:
            edges.append((tag_to_index[n1], tag_to_index[n2]))

    return np.array(edges)


def build_adjacency(edges, num_nodes):
    adjacency = [[] for _ in range(num_nodes)]

    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    return adjacency


def order_edges(start_index, adjacency):
    visited_edges = set()
    ordered_edges = []

    current = start_index
    prev = None

    while True:
        neighbors = adjacency[current]

        next_node = None
        for n in neighbors:
            edge = tuple(sorted((current, n)))
            if edge not in visited_edges:
                next_node = n
                visited_edges.add(edge)
                break

        if next_node is None:
            break

        ordered_edges.append((current, next_node))
        prev = current
        current = next_node

    return np.array(ordered_edges)


def order_nodes_down_left(coords: NDArray, down_right: bool) -> NDArray:
    if down_right:
        coords = np.flipud(coords)
    return coords


def compute_left_right_vector(
    coords: NDArray, 
    shift: int = 1
) -> Tuple[NDArray, NDArray]:
    NODE_AXES = 0
    get_left_neighbor_coords = -shift
    get_right_neighbor_coords = shift
    # Get the left and right endpoints.
    coords_right = np.roll(coords, get_right_neighbor_coords, NODE_AXES)
    coords_left  = np.roll(coords, get_left_neighbor_coords , NODE_AXES)
    # Compute the left and right vectors.
    vector_left  = coords_left  - coords
    vector_right = coords_right - coords
    return vector_left, vector_right


def compute_lengths(
    vector_left: NDArray,
    vector_right: NDArray,
    left_edges: bool = True
) -> NDArray:
    COORD_AXIS = -1
    EUCLIDEAN = 2.0
    vector = vector_left if left_edges else vector_right
    return np.linalg.norm(vector, ord=EUCLIDEAN, axis=COORD_AXIS)



def compute_inner_angle(
    vector_left: NDArray, 
    vector_right: NDArray,
    conjugate_angle: bool = False
) -> NDArray:
    if conjugate_angle:
        inner_angle = compute_angle(vector_right, vector_left)
    else:
        inner_angle = compute_angle(vector_left, vector_right)
    return inner_angle


def compute_inner_angle_sum(
    coords: NDArray,
    inner_angles: NDArray,
    sum_inner_angles: int = 3,
    conjugate_angle: bool = False
) -> NDArray:
    DECREMENT_SHIFT = -1
    SHIFT_LOWER_BOUND = 1
    # Initialize the result.
    inner_angles_sum = np.copy(inner_angles)
    # 
    for shift in range(sum_inner_angles, SHIFT_LOWER_BOUND, DECREMENT_SHIFT):
        vector_left, vector_right = compute_left_right_vector(coords, shift)
        angles = \
            compute_inner_angle(vector_left, vector_right, conjugate_angle)
        inner_angles_sum += angles
    return inner_angles_sum


def compute_angle(vector_left: NDArray, vector_right: NDArray) -> NDArray:
    ANGLE_MIN = 0.0
    ANGLE_MAX = 2 * np.pi
    # Get the X and Y coordinates of the two vectors.
    vector_left_x , vector_left_y  = vector_left.T
    vector_right_x, vector_right_y = vector_right.T
    # Compute the cross and inner product of the vectors.
    cross = vector_left_x * vector_right_y - vector_left_y * vector_right_x
    inner = vector_left_x * vector_right_x + vector_left_y * vector_right_y
    # Compute and convert the angles into range [0, 2*pi].
    angle = np.arctan2(cross, inner)
    angle = np.where(angle <= ANGLE_MIN, angle + ANGLE_MAX, angle)
    return angle


def compute_reference_length_sum(
    lengths: NDArray, 
    num_left_lengths: int = 3, 
    num_right_lengths: int = 3,
    lengths_are_left: bool = True
) -> NDArray:
    """For every reference node computes the 

    Args:
        lengths (NDArray): The array of length values.
        num_left_lengths (int, optional): The number of included left edges.
            Defaults to 3.
        num_right_lengths (int, optional): The number of included right edges.
            Defaults to 3.
        lengths_are_left (bool, optional): Indicates whether the values in
            `lengths` are for left edges. Defaults to True.

    Returns:
        NDArray: _description_
    """
    WINDOW_AXIS = -1
    # Compute how many lefts to pad with.
    from_tail = slice(-num_left_lengths + int(lengths_are_left), None)
    from_head = slice(num_right_lengths - int(not lengths_are_left))
    # 1. Pad the left with the last 'n' elements 
    #    and the right with the first 'm' elements
    padded = np.concatenate([lengths[from_tail], lengths, lengths[from_head]])
    # 2. Compute the kernel size.
    window_size = num_left_lengths + num_right_lengths
    window = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    #
    assert len(window) == len(lengths), \
        f"Window length ({len(window)}) != lengths size ({len(lengths)})"
    #
    return np.sum(window, axis=WINDOW_AXIS)


def export_boundary_info(
    text_file: str,
    coords: NDArray,
    down_right: bool = True,
    left_edges: bool = True,
    conjugate_angle: bool = False,
    sum_inner_angles: int = 3,
    sum_left_lengths: int = 3, 
    sum_right_lengths: int = 3,
    fixed_nodes: bool = True
) -> NDArray:
    """_summary_

    The first node in the export is the right most node.

    Args:
        text_file (str): _description_
        coords (NDArray): The initial node coordinates on the boundary. Has 
            shape (initial_number_of_nodes, 2).
        down_right (bool, optional): Indicates whether the right neighbor node 
            is located below the current node. Defaults to True.
        left_edges (bool, optional): _description_. Defaults to True.
        conjugate_angle (bool, optional): Indicates whether to conjugate the 
            calculated inner angles. Defaults to False.
        sum_inner_angles (int, optional): The number of inner angles to include 
            in the total reference angle summation. Defaults to 3.
        sum_left_lengths (int, optional): The number of left edges to include 
            in the total reference length summation. Defaults to 3.
        sum_right_lengths (int, optional): The number of right edges to include 
            in the total reference length summation. Defaults to 3.
        fixed_nodes (bool, optional): _description_. Defaults to True.

    Returns:
        NDArray: _description_
    """
    coords = order_nodes_down_left(coords, down_right)

    vector_left, vector_right = compute_left_right_vector(coords)

    lengths = compute_lengths(vector_left, vector_right, left_edges)

    inner_angle = compute_inner_angle(
        vector_left, vector_right, conjugate_angle
    )

    inner_angle_sum = compute_inner_angle_sum(
        coords, inner_angle, sum_inner_angles, conjugate_angle
    )

    lengths_sum = compute_reference_length_sum(
        lengths, sum_left_lengths, sum_right_lengths, left_edges
    )

    # Compute the total length of the boundary.
    boundary_length = np.sum(lengths)
    #
    with open(text_file, "w") as file:
        # Write the starting metadata
        file.write(f"{boundary_length} {len(coords)}\n")
        # Write the boundary information.
        info = np.concatenate(
            (coords, inner_angle[..., None], inner_angle_sum[..., None], lengths[..., None], lengths_sum[..., None]),
            axis=1
        )
        np.savetxt(file, info, fmt='%.18f', delimiter=" ")
        #
        if fixed_nodes:
            np.savetxt(file, coords, fmt='%.18f', delimiter=" ")



def main(
    mesh_file: str, 
    text_file: str, 
    start_node_tag: int = 1,
    down_right: bool = True,
    left_edges: bool = True,
    conjugate_angle: bool = False,
    sum_inner_angles: int = 3,
    sum_left_lengths: int = 3, 
    sum_right_lengths: int = 3,
    fixed_nodes: bool = True
) -> None:
    gmsh.initialize()
    gmsh.open(mesh_file)

    nodeTags, coords, tag_to_index = load_mesh_nodes()

    if start_node_tag not in tag_to_index:
        raise ValueError(f"Start node tag {start_node_tag} not found.")

    start_index = tag_to_index[start_node_tag]

    edges = load_line_elements(tag_to_index)

    adjacency = build_adjacency(edges, len(coords))

    ordered_edges = order_edges(start_index, adjacency)

    print("Node sequence")
    print(ordered_edges[..., 0])

    boundary = coords[ordered_edges[..., 0]]

    gmsh.finalize()

    export_boundary_info(
        text_file, 
        boundary, 
        down_right,
        left_edges,
        conjugate_angle,
        sum_inner_angles,
        sum_left_lengths, 
        sum_right_lengths,
        fixed_nodes
    )


if __name__ == "__main__":

    config_file = r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\config\unit_circle.json"

    # 0: Load the config file.
    with open(config_file, "r") as file:
        config: Dict = json.load(file)

        main(**config)