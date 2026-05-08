from typing import Union, Optional, Dict, List

# from mesh_som import MeshSom
from mesh_som_dis import Origin, MeshSom

import math
import json
import gmsh
import numpy as np
from pathlib import Path


def project_grid_boundary_from_step(
    step_path: str | Path,
    grid: np.ndarray,
    z0: float = 0.0
) -> np.ndarray:
    """
    Project boundary points of a 2D grid onto the boundary curves
    of a surface defined in a STEP file.

    Boundary points are defined as valid grid points that have at
    least one invalid (NaN) or out-of-bounds neighbor.
    """

    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(step_path)

    if grid.ndim != 3 or grid.shape[2] != 2:
        raise ValueError("grid must have shape (n, m, 2)")

    n, m, _ = grid.shape
    projected = grid.copy()

    # ------------------------------------------------------------
    # Helper: detect valid nodes
    # ------------------------------------------------------------
    valid = ~np.isnan(grid[..., 0]) & ~np.isnan(grid[..., 1])

    def neighbors(i, j):
        return [
            (i - 1, j - 1), # Top left
            (i - 1, j),     # Top
            (i - 1, j + 1), # Top right
            (i, j - 1),     # Right
            (i, j + 1),     # Left
            (i + 1, j - 1), # Bottom left
            (i + 1, j),     # Bottom
            (i + 1, j + 1)  # Bottom right
        ]

    # ------------------------------------------------------------
    # Determine effective boundary indices
    # ------------------------------------------------------------
    boundary_idx = []

    for i in range(n):
        for j in range(m):
            if not valid[i, j]:
                continue

            for ni, nj in neighbors(i, j):
                if (
                    ni < 0 or ni >= n or
                    nj < 0 or nj >= m or
                    not valid[ni, nj]
                ):
                    boundary_idx.append((i, j))
                    break

    if not boundary_idx:
        print("Warning: No boundary points detected for projection.")
        return projected

    # ------------------------------------------------------------
    # Initialize Gmsh
    # ------------------------------------------------------------
    gmsh.initialize()
    gmsh.model.add("grid_boundary_projection")

    try:
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.getEntities(dim=2)
        if not surfaces:
            raise RuntimeError("No surfaces found in STEP file")

        surface = surfaces[0]

        boundary = gmsh.model.getBoundary(
            [surface],
            oriented=False,
            recursive=False
        )

        boundary_curves = [e for e in boundary if e[0] == 1]
        if not boundary_curves:
            raise RuntimeError("No boundary curves found on surface")

        # ------------------------------------------------------------
        # Project detected boundary nodes
        # ------------------------------------------------------------
        for i, j in boundary_idx:
            x, y = grid[i, j]
            px, py, pz = x, y, z0

            best_dist = float("inf")
            best_proj = None

            for dim, tag in boundary_curves:
                closest, _ = gmsh.model.getClosestPoint(
                    dim, tag, [px, py, pz]
                )

                cx, cy, cz = closest
                dist = math.dist((px, py, pz), (cx, cy, cz))

                if dist < best_dist:
                    best_dist = dist
                    best_proj = (cx, cy)

            if best_proj is not None:
                projected[i, j] = best_proj

    finally:
        gmsh.finalize()

    return projected


def _extract_points_from_step(
    step_path: str | Path, 
    z0: float = 0.0,
    skip_tags: Optional[List[int]] = None
) -> np.ndarray:
    """Return Nx2 array of (x, y) coordinates of OCC vertices."""
    gmsh.initialize()
    try:
        gmsh.model.add("model")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        pts = []
        # dimension 0 = vertices
        for dim, tag in gmsh.model.getEntities(0):

            if skip_tags is not None and tag in skip_tags:
                continue

            x, y, z = gmsh.model.getValue(dim, tag, [])
            # if you prefer, you could check |z - z0| here
            pts.append([x, y])

        if not pts:
            return np.empty((0, 2))

        return np.asarray(pts, dtype=float)
    finally:
        gmsh.finalize()


def project_closest_node_on_point(
    step_path: str | Path,
    grid: np.ndarray,
    z0: float = 0.0,
    skip_tags: Optional[List[int]] = None
) -> np.ndarray:
    """
    For each OCC vertex from the STEP geometry, find the closest
    valid node in `grid` and overwrite that node with the point coordinates.

    Parameters
    ----------
    step_path : path to STEP file
    grid : ndarray (nr, nc, 2)
        NaN entries are ignored.
    z0 : float
        Optional reference z (currently unused, but kept for API compatibility).

    Returns
    -------
    ndarray : modified copy of grid
    """
    new_grid = np.copy(grid)

    # Extract points from STEP
    points = _extract_points_from_step(step_path, z0, skip_tags)
    if len(points) == 0:
        return new_grid
    
    print("Keept tags", points)

    # valid nodes = both coordinates finite
    valid_mask = np.isfinite(new_grid[..., 0]) & np.isfinite(new_grid[..., 1])
    if not np.any(valid_mask):
        return new_grid

    # flatten for vector distance computation
    valid_indices = np.argwhere(valid_mask)  # (K, 2)
    nodes = new_grid[valid_mask]  # (K, 2)

    for pt in points:
        dists = np.linalg.norm(nodes - pt, axis=1)
        k = np.argmin(dists)
        r, c = valid_indices[k]
        new_grid[r, c] = pt

    return new_grid


def sample_step_geometry(
    step_file_path: str,
    num_curve_samples: int,
    num_surface_samples: int,
    max_sample_dimension: int = 2,
    separate_curve_samples: bool = False,
    num_row_surface_samples: Optional[int] = None,
    num_col_surface_samples: Optional[int] = None,
    tag_curve_samples: Optional[Dict[int, int]] = None,
):
    """
    Reads a STEP file, samples surfaces first and then their boundary curves.

    Args:
        step_file_path (str): Path to the input STEP file.
        num_curve_samples (int): Default number of linearly spaced samples per curve.
        num_surface_samples (int): Default number of samples per surface direction.
        max_sample_dimension (int): Number of coordinate components returned
                                    per point (1=x, 2=xy, 3=xyz).
        separate_curve_samples (bool): If True, return a list with one array per
                                       curve. Otherwise concatenate.
        num_row_surface_samples (Optional[int]): Override number of samples
                                    in the first parametric direction.
        num_col_surface_samples (Optional[int]): Override number of samples
                                    in the second parametric direction.
        tag_curve_samples (Optional[Dict[int, int]]): Map curve tag -> number
                                    of samples. Falls back to num_curve_samples.

    Returns:
        tuple:
            curve samples, surface samples
    """
    if max_sample_dimension < 1 or max_sample_dimension > 3:
        raise ValueError("max_sample_dimension must be 1, 2 or 3.")

    gmsh.initialize()
    gmsh.open(step_file_path)

    all_curve_samples: List[np.ndarray] = []
    all_surface_samples: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # 1) SAMPLE SURFACES FIRST
    # ------------------------------------------------------------------
    surfaces = gmsh.model.getEntities(dim=2)
    print(f"Found {len(surfaces)} surfaces.", "info")

    # choose resolution
    n_rows = (
        num_row_surface_samples
        if num_row_surface_samples is not None
        else num_surface_samples
    )
    n_cols = (
        num_col_surface_samples
        if num_col_surface_samples is not None
        else num_surface_samples
    )

    for dim, tag in surfaces:
        try:
            uv_bound = gmsh.model.getParametrizationBounds(dim, tag)
            u_min, v_min, u_max, v_max = np.array(uv_bound).flatten()

            u_coords = np.linspace(u_min, u_max, n_rows)
            v_coords = np.linspace(v_min, v_max, n_cols)

            U, V = np.meshgrid(u_coords, v_coords)
            parametric_coords = np.stack((U.flatten(), V.flatten()), axis=-1)

            inside_points = np.zeros((parametric_coords.shape[0],), dtype=bool)

            for i, point_2d in enumerate(parametric_coords):
                inside_points[i] = (
                    gmsh.model.isInside(dim, tag, point_2d, parametric=True) > 0
                )

            surface_points = gmsh.model.getValue(
                dim, tag, parametric_coords.flatten()
            ).reshape(-1, 3)

            surface_points = surface_points[inside_points, :]
            all_surface_samples.append(surface_points)

        except Exception as e:
            print(f"Error sampling surface {tag}: {e}", "error")

    # ------------------------------------------------------------------
    # 2) GET CURVES FROM SURFACE BOUNDARIES
    # ------------------------------------------------------------------
    boundary_dimtags = gmsh.model.getBoundary(
        surfaces, combined=True, oriented=False, recursive=False
    )
    curves = list({(d, abs(t)) for d, t in boundary_dimtags if d == 1})

    print(f"Sampling {len(curves)} boundary curves.", "info")


    if tag_curve_samples is not None:
        tag_curve_samples = \
            { int(key): int(val) for key, val in tag_curve_samples.items() }

    for dim, tag in curves:
        try:
            # choose per-curve resolution
            if tag_curve_samples is not None and tag in tag_curve_samples:
                n_curve = tag_curve_samples[tag]
            else:
                n_curve = num_curve_samples

            min_u, max_u = gmsh.model.getParametrizationBounds(dim, tag)
            parametric_coords = np.linspace(min_u, max_u, n_curve).flatten()

            curve_points = gmsh.model.getValue(
                dim, tag, parametric_coords
            ).reshape(-1, 3)

            all_curve_samples.append(curve_points)

        except Exception as e:
            print(f"Error sampling curve {tag}: {e}", "error")

    gmsh.finalize()

    # ------------------------------------------------------------------
    # 3) FORMAT OUTPUT
    # ------------------------------------------------------------------
    if separate_curve_samples:
        curve_samples_array = [
            c[:, :max_sample_dimension] for c in all_curve_samples
        ]
    else:
        if all_curve_samples:
            curve_samples_array = (
                np.vstack(all_curve_samples)[:, :max_sample_dimension]
            )
        else:
            curve_samples_array = np.array([]).reshape(-1, max_sample_dimension)

    if all_surface_samples:
        surface_samples_array = (
            np.vstack(all_surface_samples)[:, :max_sample_dimension]
        )
    else:
        surface_samples_array = np.array([]).reshape(-1, max_sample_dimension)

    return curve_samples_array, surface_samples_array



# def sample_step_geometry(
#     step_file_path: str, 
#     num_curve_samples: int, 
#     num_surface_samples: int,
#     max_sample_dimension: int = 2,
#     separate_curve_samples: bool = False,
# ):
#     """
#     Reads a STEP file, samples all curves and surfaces with linearly spaced
#     parametrization, and returns the samples in two NumPy arrays.

#     Args:
#         step_file_path (str): Path to the input STEP file.
#         num_curve_samples (int): Number of linearly spaced samples per curve.
#         num_surface_samples (int): Number of linearly spaced samples per surface
#                                    along each parametric direction (u and v).

#     Returns:
#         tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
#                                         - curve_samples: An array of shape (-1, 3)
#                                                          containing (x, y, z) coordinates
#                                                          for all curve samples.
#                                         - surface_samples: An array of shape (-1, 3)
#                                                            containing (x, y, z) coordinates
#                                                            for all surface samples.
#     """

#     gmsh.initialize()
#     gmsh.open(step_file_path)

#     all_curve_samples = []
#     all_surface_samples = []

#     # Get all curves (dim = 1)
#     curves = gmsh.model.getEntities(dim=1)
#     print(f"Found {len(curves)} curves.", "info")

#     for dim, tag in curves:
#         try:
#             # Get parametric bounds for the curve
#             min_u, max_u = gmsh.model.getParametrizationBounds(dim, tag)
#             parametric_coords = \
#                 np.linspace(min_u, max_u, num_curve_samples).flatten()

#             # Sample the curve
#             curve_points = gmsh.model.getValue(dim, tag, parametric_coords)

#             # Reshape to (num_samples, 3)
#             curve_points_reshaped = np.array(curve_points).reshape(-1, 3)
#             all_curve_samples.append(curve_points_reshaped)
#         except Exception as e:
#             print(f"Error sampling curve {tag}: {e}", "error")

#     # Get all surfaces (dim = 2)
#     surfaces = gmsh.model.getEntities(dim=2)
#     print(f"Found {len(surfaces)} surfaces.", "info")

#     for dim, tag in surfaces:
#         try:
#             # Get parametric bounds for the surface (u_min, u_max, v_min, v_max)
#             uv_bound = gmsh.model.getParametrizationBounds(dim, tag)

#             u_min, v_min, u_max, v_max = np.array(uv_bound).flatten()

#             u_coords = np.linspace(u_min, u_max, num_surface_samples).flatten()
#             v_coords = np.linspace(v_min, v_max, num_surface_samples).flatten()

#             # Create a meshgrid for u and v
#             U, V = np.meshgrid(u_coords, v_coords)

#             # Flatten U and V and interleave them for getValue
#             parametric_coords = \
#                 np.stack((U.flatten(), V.flatten()), axis=-1).reshape(-1, 2)
            
#             inside_points = np.zeros((parametric_coords.shape[0], ), dtype=np.bool)
            
#             for i in range(parametric_coords.shape[0]):
#                 point_2d = parametric_coords[i, :]

#                 inside_points[i] = \
#                     gmsh.model.isInside(dim, tag, point_2d, parametric=True) > 0

#             # Sample the surface
#             surface_points = \
#                 gmsh.model.getValue(dim, tag, parametric_coords.flatten()).reshape(-1, 3)
            
#             surface_points = surface_points[inside_points, :]

#             all_surface_samples.append(surface_points)

#         except Exception as e:
#             print(f"Error sampling surface {tag}: {e}", "error")

#     gmsh.finalize()

#     # Concatenate all curve samples
#     if all_curve_samples:
#         curve_samples_array = np.vstack(all_curve_samples)
#     else:
#         curve_samples_array = np.array([]).reshape(-1, 3)

#     # Concatenate all surface samples
#     if all_surface_samples:
#         surface_samples_array = np.vstack(all_surface_samples)
#     else:
#         surface_samples_array = np.array([]).reshape(-1, 3)

#     return curve_samples_array[:,:2], surface_samples_array[:,:2]


def _almost_collinear(p_prev, p_mid, p_next, tol):
    """
    Return True if p_mid is almost on the segment p_prev -> p_next.
    """
    v1 = p_mid - p_prev
    v2 = p_next - p_prev
    area2 = abs(v1[0] * v2[1] - v1[1] * v2[0])  # 2 * triangle area
    scale = np.linalg.norm(v2)
    if scale == 0:
        return True
    return area2 / scale < tol


def quad_mesh_on_step_surface(
    points,
    step_file,
    msh_file="mesh.msh",
    collapse_collinear=False,
    collinear_tol=1e-8,
) -> None:
    """
    Attach a quadrilateral mesh defined by a (n, m, 2) NumPy array
    to a surface imported from a STEP file.

    Points containing NaN values are excluded from the mesh.
    Quadrilateral elements are only created if all four corner
    nodes are present.

    If collapse_collinear=True, a quad is replaced by a triangle
    whenever one vertex is almost collinear with its neighbors.
    """

    n, m, _ = points.shape

    gmsh.initialize()
    gmsh.model.add("quad_mesh_from_array")

    # ------------------------------------------------------------
    # 1. Import STEP surface
    # ------------------------------------------------------------
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    if len(surfaces) == 0:
        raise RuntimeError("No surfaces found in STEP file.")

    surface_tag = surfaces[0][1]

    # ------------------------------------------------------------
    # 2. Add nodes (skip NaNs)
    # ------------------------------------------------------------
    node_map = {}
    coords = []
    node_tags = []

    next_node_tag = 1

    for i in range(n):
        for j in range(m):
            x, y = points[i, j]
            if np.isnan(x) or np.isnan(y):
                continue

            node_map[(i, j)] = next_node_tag
            node_tags.append(next_node_tag)
            coords.extend([x, y, 0.0])
            next_node_tag += 1

    if not node_tags:
        raise RuntimeError("No valid (non-NaN) points found.")

    gmsh.model.mesh.addNodes(
        dim=2,
        tag=surface_tag,
        nodeTags=node_tags,
        coord=coords
    )

    # ------------------------------------------------------------
    # 3. Create elements
    # ------------------------------------------------------------
    quad_tags, quad_nodes = [], []
    tri_tags, tri_nodes = [], []

    elem_tag = 1

    for i in range(n - 1):
        for j in range(m - 1):
            corners_ij = [
                (i, j),
                (i + 1, j),
                (i + 1, j + 1),
                (i, j + 1),
            ]

            if not all(c in node_map for c in corners_ij):
                continue

            # normal quad mode
            if not collapse_collinear:
                quad_tags.append(elem_tag)
                for c in corners_ij:
                    quad_nodes.append(node_map[c])
                elem_tag += 1
                continue

            # ----------------------------------------------------
            # collapse logic
            # ----------------------------------------------------
            pts = [points[c] for c in corners_ij]

            # check each vertex as potential middle point
            drop_index = None
            for k in range(4):
                p_prev = pts[(k - 1) % 4]
                p_mid = pts[k]
                p_next = pts[(k + 1) % 4]

                if _almost_collinear(p_prev, p_mid, p_next, collinear_tol):
                    drop_index = k
                    break

            if drop_index is None:
                # keep quad
                quad_tags.append(elem_tag)
                for c in corners_ij:
                    quad_nodes.append(node_map[c])
            else:
                # emit triangle without that node
                tri_tags.append(elem_tag)
                for k, c in enumerate(corners_ij):
                    if k != drop_index:
                        tri_nodes.append(node_map[c])

            elem_tag += 1

    # ------------------------------------------------------------
    # 4. Push elements to gmsh
    # ------------------------------------------------------------
    if quad_tags:
        gmsh.model.mesh.addElements(
            dim=2,
            tag=surface_tag,
            elementTypes=[3],  # quad
            elementTags=[quad_tags],
            nodeTags=[quad_nodes],
        )

    if tri_tags:
        gmsh.model.mesh.addElements(
            dim=2,
            tag=surface_tag,
            elementTypes=[2],  # triangle
            elementTags=[tri_tags],
            nodeTags=[tri_nodes],
        )

    if not quad_tags and not tri_tags:
        print("Warning: No elements were created.")

    # ------------------------------------------------------------
    # 5. Write mesh
    # ------------------------------------------------------------
    gmsh.write(msh_file)
    gmsh.finalize()

    print(f"Mesh written to {msh_file}")


# def quad_mesh_on_step_surface(points, step_file, msh_file="mesh.msh") -> None:
#     """
#     Attach a quadrilateral mesh defined by a (n, m, 2) NumPy array
#     to a surface imported from a STEP file.

#     Points containing NaN values are excluded from the mesh.
#     Quadrilateral elements are only created if all four corner
#     nodes are present.
#     """

#     n, m, _ = points.shape

#     gmsh.initialize()
#     gmsh.model.add("quad_mesh_from_array")

#     # ------------------------------------------------------------
#     # 1. Import STEP surface
#     # ------------------------------------------------------------
#     gmsh.model.occ.importShapes(step_file)
#     gmsh.model.occ.synchronize()

#     surfaces = gmsh.model.getEntities(dim=2)
#     if len(surfaces) == 0:
#         raise RuntimeError("No surfaces found in STEP file.")

#     surface_tag = surfaces[0][1]

#     # ------------------------------------------------------------
#     # 2. Add nodes (skip NaNs)
#     # ------------------------------------------------------------
#     node_map = {}        # (i, j) -> node_tag
#     coords = []
#     node_tags = []

#     next_node_tag = 1

#     for i in range(n):
#         for j in range(m):
#             x, y = points[i, j]
#             if np.isnan(x) or np.isnan(y):
#                 continue

#             node_map[(i, j)] = next_node_tag
#             node_tags.append(next_node_tag)
#             coords.extend([x, y, 0.0])
#             next_node_tag += 1

#     if not node_tags:
#         raise RuntimeError("No valid (non-NaN) points found.")

#     gmsh.model.mesh.addNodes(
#         dim=2,
#         tag=surface_tag,
#         nodeTags=node_tags,
#         coord=coords
#     )

#     # ------------------------------------------------------------
#     # 3. Create quadrilateral elements (only if all nodes exist)
#     # ------------------------------------------------------------
#     quad_tags = []
#     quad_nodes = []

#     elem_tag = 1

#     for i in range(n - 1):
#         for j in range(m - 1):
#             corners = [
#                 (i, j),
#                 (i + 1, j),
#                 (i + 1, j + 1),
#                 (i, j + 1)
#             ]

#             # Check if all four nodes exist
#             if all(corner in node_map for corner in corners):
#                 quad_tags.append(elem_tag)
#                 for corner in corners:
#                     quad_nodes.append(node_map[corner])
#                 elem_tag += 1

#     if quad_tags:
#         gmsh.model.mesh.addElements(
#             dim=2,
#             tag=surface_tag,
#             elementTypes=[3],        # 4-node quadrilateral
#             elementTags=[quad_tags],
#             nodeTags=[quad_nodes]
#         )
#     else:
#         print("Warning: No quadrilateral elements were created.")

#     # ------------------------------------------------------------
#     # 4. Write mesh
#     # ------------------------------------------------------------
#     gmsh.write(msh_file)
#     gmsh.finalize()

#     print(f"Mesh written to {msh_file}")


def tri_mesh_on_step_surface(
    points: np.ndarray,
    step_file: str | Path,
    msh_file: str = "mesh.msh"
) -> None:
    """
    Attach a triangular mesh defined by a (n, m, 2) NumPy array
    to a surface imported from a STEP file.

    Points containing NaN values are excluded from the mesh.
    Triangles are only created if all three corner nodes exist.
    """

    points = np.asarray(points)
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError("points must have shape (n, m, 2)")

    n, m, _ = points.shape

    gmsh.initialize()
    gmsh.model.add("tri_mesh_from_array")

    # ------------------------------------------------------------
    # 1. Import STEP surface
    # ------------------------------------------------------------
    gmsh.model.occ.importShapes(str(step_file))
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    if not surfaces:
        raise RuntimeError("No surfaces found in STEP file.")

    surface_tag = surfaces[0][1]

    # ------------------------------------------------------------
    # 2. Add nodes (skip NaNs)
    # ------------------------------------------------------------
    node_map = {}        # (i, j) -> node_tag
    node_tags = []
    coords = []

    next_node_tag = 1

    for i in range(n):
        for j in range(m):
            x, y = points[i, j]
            if np.isnan(x) or np.isnan(y):
                continue

            node_map[(i, j)] = next_node_tag
            node_tags.append(next_node_tag)
            coords.extend([x, y, 0.0])
            next_node_tag += 1

    if not node_tags:
        raise RuntimeError("No valid (non-NaN) points found.")

    gmsh.model.mesh.addNodes(
        dim=2,
        tag=surface_tag,
        nodeTags=node_tags,
        coord=coords
    )

    # ------------------------------------------------------------
    # 3. Create triangular elements
    # ------------------------------------------------------------
    tri_tags = []
    tri_nodes = []

    elem_tag = 1

    for i in range(n - 1):
        for j in range(m - 1):
            p00 = (i, j)
            p10 = (i + 1, j)
            p11 = (i + 1, j + 1)
            p01 = (i, j + 1)

            # Triangle 1: p00, p10, p11
            if all(p in node_map for p in (p00, p10, p11)):
                tri_tags.append(elem_tag)
                tri_nodes.extend([
                    node_map[p00],
                    node_map[p10],
                    node_map[p11]
                ])
                elem_tag += 1

            # Triangle 2: p00, p11, p01
            if all(p in node_map for p in (p00, p11, p01)):
                tri_tags.append(elem_tag)
                tri_nodes.extend([
                    node_map[p00],
                    node_map[p11],
                    node_map[p01]
                ])
                elem_tag += 1

    if tri_tags:
        gmsh.model.mesh.addElements(
            dim=2,
            tag=surface_tag,
            elementTypes=[2],   # 3-node triangle
            elementTags=[tri_tags],
            nodeTags=[tri_nodes]
        )
    else:
        print("Warning: No triangular elements were created.")

    # ------------------------------------------------------------
    # 4. Write mesh
    # ------------------------------------------------------------
    gmsh.write(str(msh_file))
    gmsh.finalize()

    print(f"Triangular mesh written to {msh_file}")


ORIGIN_MAP = {
    "TOP_LEFT": Origin.TOP_LEFT,
    "TOP_RIGHT": Origin.TOP_RIGHT,
    "BOTTOM_LEFT": Origin.BOTTOM_LEFT,
    "BOTTOM_RIGHT": Origin.BOTTOM_RIGHT,
}


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
    

def resolve_geometry_data(geo_cfg: dict):
    """
    Returns (curve_data, surface_data) based on the geometry configuration.

    Priority:
      1) Explicit curve_data / surface_data (inline JSON)
      2) STEP-based sampling
    """

    curve_data_raw = geo_cfg.get("curve_data")
    surface_data_raw = geo_cfg.get("surface_data")

    # Case 1: Explicit data provided in JSON
    if curve_data_raw is not None or surface_data_raw is not None:
        if curve_data_raw is None or surface_data_raw is None:
            raise ValueError(
                "Both 'curve_data' and 'surface_data' must be provided together."
            )

        curve_data = np.asarray(curve_data_raw, dtype=np.float64)
        surface_data = np.asarray(surface_data_raw, dtype=np.float64)

        # Basic shape validation
        if curve_data.ndim != 2 or curve_data.shape[1] != 2:
            raise ValueError("'curve_data' must have shape (N, 2).")

        if surface_data.ndim != 2 or surface_data.shape[1] != 2:
            raise ValueError("'surface_data' must have shape (M, 2).")

        return curve_data, surface_data

    # Case 2: Sample from STEP file
    if geo_cfg.get("step_file") is None:
        raise ValueError(
            "Either STEP geometry or explicit curve/surface data must be provided."
        )

    return sample_step_geometry(
        geo_cfg["step_file"],
        geo_cfg["curve_samples_per_entity"],
        geo_cfg["surface_samples_per_entity"],
        geo_cfg.get("max_sample_dimension", 2),
        geo_cfg.get("separate_curve_samples", False),
        geo_cfg.get("num_row_surface_samples", None),
        geo_cfg.get("num_col_surface_samples", None),
        geo_cfg.get("tag_curve_samples", None)
    )


def create_som(cfg: dict) -> MeshSom:
    som_cfg = cfg["som"]

    return MeshSom(
        row_dimension=som_cfg["row_dimension"],
        col_dimension=som_cfg["col_dimension"],
        input_dimension=som_cfg.get("input_dimension", 2),
        sigma=som_cfg.get("sigma", 1.0),
        learning_rate=som_cfg.get("learning_rate", 0.5),
        random_seed=som_cfg.get("random_seed", None),
    )


def initialize_weights(som: MeshSom, cfg: dict):
    init_cfg = cfg["initialization"]

    # Base init
    if init_cfg["type"] == "grid":
        g = init_cfg["grid"]
        som.grid_weights_init(
            x_min=g["x_min"],
            x_max=g["x_max"],
            y_min=g["y_min"],
            y_max=g["y_max"],
        )

        # Randomize interior
        if init_cfg.get("randomize_interior", False):
            r, c = som._map_shape
            som._weights[1:-1, 1:-1] = np.random.rand(r - 2, c - 2, 2)

    elif init_cfg["type"] == "data":
        data = np.asarray(init_cfg["data"])
        origin = ORIGIN_MAP[init_cfg["origin"]]
        som.data_weights_init(data, origin)

    else:
        raise ValueError(f"Unknown initialization type: {init_cfg['type']}")


def apply_fixed_weights(som: MeshSom, cfg: dict):
    fw_cfg = cfg.get("fixed_weights", {})
    if not fw_cfg.get("enabled", False):
        return

    origin = ORIGIN_MAP[fw_cfg["origin"]]

    if fw_cfg.get("boundary_only", False):
        mask = np.ones(som._map_shape, dtype=bool)
        mask[1:-1, 1:-1] = False
    else:
        mask = np.asarray(fw_cfg["mask"], dtype=bool)

    som.fix_weights(mask, origin)


def apply_disabled_weights(som: MeshSom, cfg: dict):
    dw_cfg = cfg.get("disabled_weights", {})
    if not dw_cfg.get("enabled", False):
        return

    origin = ORIGIN_MAP[dw_cfg["origin"]]
    mask = np.asarray(dw_cfg["mask"], dtype=bool)

    som.disable_weights(mask, origin)


def train_som(
    som: MeshSom,
    curve_data: np.ndarray | list[np.ndarray],
    surface_data: np.ndarray,
    complete_data: np.ndarray,
    cfg: dict,
):
    train_cfg = cfg["training"]
    method = train_cfg["method"]

    if method == "composite":
        som.train_composite(
            max_iteration=train_cfg["max_iteration"],
            initial_final_iter=train_cfg["initial_final_iter"],
            boundary_data=curve_data,
            interior_data=surface_data,
            complete_data=complete_data,
            random_order=train_cfg.get("random_order", True),
            verbose=train_cfg.get("verbose", False),
        )

    elif method == "multiconnected":
        if not isinstance(curve_data, list):
            raise ValueError(
                "For 'multiconnected' training, curve_data must be a list of arrays."
            )

        groups = train_cfg.get("boundary_groups")
        if groups is None:
            raise ValueError(
                "'boundary_groups' must be provided in the config for "
                "'multiconnected' training."
            )

        if len(groups) != len(curve_data):
            raise ValueError(
                f"'boundary_groups' must have the same length {len(groups)} " \
                f"as curve_data {len(curve_data)}."
            )

        # --------------------------------------------------------------
        # stack curves according to group id
        # --------------------------------------------------------------
        from collections import defaultdict

        grouped = defaultdict(list)
        for gid, data in zip(groups, curve_data):
            grouped[gid].append(data)

        # sort by group id to get deterministic ordering
        stacked_groups = [
            np.vstack(grouped[gid]) for gid in sorted(grouped.keys())
        ]

        if len(stacked_groups) == 0:
            raise ValueError("No boundary groups were constructed.")

        outer_boundary_data = stacked_groups[0]
        inner_boundary_data = stacked_groups[1:]

        som.train_multiconnected(
            train_cfg["max_iteration"],
            train_cfg["initial_final_iter"],
            surface_data,
            outer_boundary_data,
            *inner_boundary_data,
            complete_data=complete_data,
            macro_first_input=train_cfg.get("macro_first_input", 0),
            macro_input_step=train_cfg.get("macro_input_step", 1),
            random_order=train_cfg.get("random_order", True),
            verbose=train_cfg.get("verbose", False),
        )

    else:
        raise ValueError(f"Unknown training method: {method}")