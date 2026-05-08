import numpy as np
from numpy.typing import NDArray


def cross2d(a: NDArray, b: NDArray) -> NDArray:
    X = (Ellipsis, 0)
    Y = (Ellipsis, 1)
    return a[X] * b[Y] - a[Y] * b[X]


def inner2d(a: NDArray, b: NDArray) -> NDArray:
    X = (Ellipsis, 0)
    Y = (Ellipsis, 1)
    return a[X] * b[X] + a[Y] * b[Y]


def aspect_ratio_warnings(
    quality: NDArray, 
    limit: float = 20.0
) -> NDArray[np.bool]:
    return quality >= limit


def aspect_ratio_error(
    quality: NDArray, 
    limit: float = 1000.0
) -> NDArray[np.bool]:
    return quality >= limit


def triangle_aspect_ratio(nodes: NDArray, tris: NDArray) -> NDArray:
    """
    Computes the triangle aspect ratio based on the Ansys interior angle definition.
    
    Args:
        nodes: (num_nodes, 2) array of coordinates.
        tris: (num_tris, 3) array of node indices.
        
    Returns:
        (num_tris,) array of aspect ratios.
    """
    # 1. Gather coordinates: (T, 3, 2)
    P = nodes[tris]
    P0, P1, P2 = P[:, 0, :], P[:, 1, :], P[:, 2, :]

    # 2. Define vectors for each corner
    # At Node 0: vectors (P1-P0) and (P2-P0)
    v01, v02 = P1 - P0, P2 - P0
    # At Node 1: vectors (P0-P1) and (P2-P1)
    v10, v12 = P0 - P1, P2 - P1
    # At Node 2: vectors (P0-P2) and (P1-P2)
    v20, v21 = P0 - P2, P1 - P2

    def get_cos_alpha(u, v):
        dot = np.sum(u * v, axis=1)
        norm_u = np.linalg.norm(u, axis=1)
        norm_v = np.linalg.norm(v, axis=1)
        # Avoid division by zero for degenerate triangles
        denom = np.maximum(norm_u * norm_v, 1e-14)
        # Clip to [-1, 1] to avoid numerical precision issues
        return np.clip(dot / denom, -1.0, 1.0)

    cos_a0 = get_cos_alpha(v01, v02)
    cos_a1 = get_cos_alpha(v10, v12)
    cos_a2 = get_cos_alpha(v20, v21)
    
    cos_alphas = np.stack([cos_a0, cos_a1, cos_a2], axis=1) # (T, 3)

    # 3. Calculate nodal aspect ratio r
    # r = 0.5 / (1 - cos) if cos >= 0.5
    # r = 1.5 / (1 + cos) if cos < 0.5
    
    # We use np.where to handle the piecewise condition
    eps = 1e-14
    r = np.where(
        cos_alphas >= 0.5,
        0.5 / np.maximum(1.0 - cos_alphas, eps),
        1.5 / np.maximum(1.0 + cos_alphas, eps)
    )

    # 4. Final computation
    mean_r = np.mean(r, axis=1)
    
    # Aspect Ratio = 2 * (sqrt(mean_r) - 1) + 1
    # Note: If mean_r is slightly less than 1 due to precision, clip to 0
    aspect_ratio = 2.0 * (np.sqrt(np.maximum(mean_r, 0.0)) - 1.0) + 1.0
    
    return aspect_ratio


def quad_aspect_ratio(nodes: NDArray, quads: NDArray) -> NDArray:
    """Compute aspect ratio using midpoint-based principal axes."""
    P = nodes[quads]  # (Q, 4, 2)

    P0 = P[:, 0, :]
    P1 = P[:, 1, :]
    P2 = P[:, 2, :]
    P3 = P[:, 3, :]

    # Midpoints
    M0 = 0.5 * (P0 + P1)
    M1 = 0.5 * (P1 + P2)
    M2 = 0.5 * (P2 + P3)
    M3 = 0.5 * (P3 + P0)

    # Principal axes
    X1 = M1 - M3
    X2 = M2 - M0

    # Helper functions
    def dot(a, b):
        return np.sum(a * b, axis=1)

    def norm(v):
        return np.linalg.norm(v, axis=1)

    def proj(u, v):
        """Projection of u onto v"""
        v_norm_sq = dot(v, v)
        v_norm_sq = np.where(v_norm_sq < 1e-14, 1e-14, v_norm_sq)
        return (dot(u, v) / v_norm_sq)[:, None] * v

    # Orthogonal components
    Y1 = X2 - proj(X2, X1)  # corrected
    Y2 = X1 - proj(X1, X2)

    # Side lengths
    a1 = norm(X1)
    b1 = norm(Y1)

    a2 = norm(X2)
    b2 = norm(Y2)

    # Aspect ratios
    def aspect(a, b):
        eps = 1e-14
        a = np.maximum(a, eps)
        b = np.maximum(b, eps)
        return np.maximum(a, b) / np.minimum(a, b)

    ar1 = aspect(a1, b1)
    ar2 = aspect(a2, b2)

    return np.maximum(ar1, ar2)


def triangle_jacobian_ratio(nodes: NDArray, tris: NDArray) -> NDArray:
    """
    Computes the triangle aspect ratio based on the Ansys interior angle definition.
    
    Args:
        nodes: (num_nodes, 2) array of coordinates.
        tris: (num_tris, 3) array of node indices.
        
    Returns:
        (num_tris,) array of aspect ratios.
    """
    BEST_POSSIBLE_JACOBIAN_RATIO = 1.0
    return np.full((len(tris), ), BEST_POSSIBLE_JACOBIAN_RATIO)


def quad_jacobian_ratio(nodes: NDArray, quads: NDArray) -> NDArray:
    # 1. Get coordinates for each node in the quads: (N, 4, 2)
    P = nodes[quads]
    
    # 2. Define vectors for the two edges meeting at each corner
    # Corner 0: vectors (P1-P0) and (P3-P0)
    v0_1 = P[:, 1] - P[:, 0]
    v0_2 = P[:, 3] - P[:, 0]
    
    # Corner 1: vectors (P2-P1) and (P0-P1)
    v1_1 = P[:, 2] - P[:, 1]
    v1_2 = P[:, 0] - P[:, 1]
    
    # Corner 2: vectors (P3-P2) and (P1-P2)
    v2_1 = P[:, 3] - P[:, 2]
    v2_2 = P[:, 1] - P[:, 2]
    
    # Corner 3: vectors (P0-P3) and (P2-P3)
    v3_1 = P[:, 0] - P[:, 3]
    v3_2 = P[:, 2] - P[:, 3]

    # 3. Compute the determinants (cross products) at each corner
    # The 0.25 scaling factor is omitted as it cancels out in the ratio
    det0 = cross2d(v0_1, v0_2)
    det1 = cross2d(v1_1, v1_2)
    det2 = cross2d(v2_1, v2_2)
    det3 = cross2d(v3_1, v3_2)
    
    dets = np.stack([det0, det1, det2, det3], axis=1)

    # 4. Compute Ratio: min / max
    min_det = np.min(dets, axis=1)
    max_det = np.max(dets, axis=1)
    
    # Avoid division by zero for degenerate elements
    ratio = np.divide(max_det, min_det)
    
    return ratio


def jacobian_ratio_warning(
    quality: NDArray,
    limit: float = 30.0
) -> NDArray[np.bool]:
    return quality >= limit


def jacobian_ratio_error(
    quality: NDArray,
    limit: float = 40.0
) -> NDArray[np.bool]:
    return quality >= limit



def quad_parallel_deviation(nodes: NDArray, quads: NDArray) -> NDArray:
    """
    Computes the Parallel Deviation for quads (max angle between opposite edges).
    """
    P = nodes[quads] # (Q, 4, dim)
    
    # Edge vectors
    # L0: P0 -> P1, L2: P3 -> P2 (Opposite pair 1)
    # L1: P1 -> P2, L3: P0 -> P3 (Opposite pair 2)
    L0 = P[:, 1] - P[:, 0]
    L2 = P[:, 2] - P[:, 3] # Vectors pointed in same direction
    
    L1 = P[:, 2] - P[:, 1]
    L3 = P[:, 3] - P[:, 0] # Vectors pointed in same direction

    def get_angle(v1, v2):
        norm1 = np.linalg.norm(v1, axis=-1)
        norm2 = np.linalg.norm(v2, axis=-1)
        
        # Unit vectors
        u1 = v1 / np.maximum(norm1, 1e-14)[..., None]
        u2 = v2 / np.maximum(norm2, 1e-14)[..., None]
        
        dot = np.sum(u1 * u2, axis=-1)
        # Clip for float precision and compute angle in degrees
        return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))

    dev1 = get_angle(L0, L2)
    dev2 = get_angle(L1, L3)

    return np.maximum(dev1, dev2)


def parallel_deviation_warning(
    quality: NDArray, 
    limit: float = 70.0
) -> NDArray[np.bool]:
    return quality >= limit


def parallel_deviation_error(
    quality: NDArray, 
    limit: float = 150.0
) -> NDArray[np.bool]:
    return quality >= limit


def triangle_parallel_deviation(nodes: NDArray, tris: NDArray) -> NDArray:
    """
    Computes the triangle aspect ratio based on the Ansys interior angle definition.
    
    Args:
        nodes: (num_nodes, 2) array of coordinates.
        tris: (num_tris, 3) array of node indices.
        
    Returns:
        (num_tris,) array of aspect ratios.
    """
    BEST_POSSIBLE_PARALLEL_DEVIATION = 0.0
    return np.full((len(tris), ), BEST_POSSIBLE_PARALLEL_DEVIATION)


def max_corner_angle(nodes: NDArray, elements: NDArray) -> NDArray:
    """
    Computes the maximum interior corner angle for triangles or quadrilaterals.
    Assumes Counter-Clockwise (CCW) node ordering.
    """
    P = nodes[elements]  # (E, num_nodes, 2)

    MIN_ANGLE = 0.0
    MAX_ANGLE = 2 * np.pi
    NODE_AXIS = 1
    GET_PREV_NODE = 1
    GET_NEXT_NODE = -1

    # Use 1 for previous node (i-1) and -1 for next node (i+1)
    prev_node = np.roll(P, GET_PREV_NODE, NODE_AXIS)
    next_node = np.roll(P, GET_NEXT_NODE, NODE_AXIS)

    # Vectors pointing AWAY from the current corner
    v_prev = prev_node - P
    v_next = next_node - P

    # 2D cross and inner products
    cross = cross2d(v_next, v_prev)
    inner = inner2d(v_next, v_prev)

    # Calculate oriented angle in range (-pi, pi]
    angles = np.arctan2(cross, inner)
    
    # Map to (0, 2*pi] to ensure we get the interior sweep for CCW elements
    # Using np.pi * 2 for the shift
    angles = np.where(angles <= MIN_ANGLE, angles + MAX_ANGLE, angles)
    
    # Convert to degrees for Ansys-consistent output
    angles = np.degrees(angles)
    
    # Return the max across the node axis (axis 1)
    return np.max(angles, axis=NODE_AXIS)


def triangle_max_corner_angle_warning(
    quality: NDArray,
    limit: float = 165.0
) -> NDArray[np.bool]:
    return quality >= limit


def triangle_max_corner_angle_error(
    quality: NDArray,
    limit: float = 179.9
) -> NDArray[np.bool]:
    return quality >= limit


def quad_max_corner_angle_warning(
    quality: NDArray,
    limit: float = 155.0
) -> NDArray[np.bool]:
    return quality >= limit


def quad_max_corner_angle_error(
    quality: NDArray,
    limit: float = 179.9
) -> NDArray[np.bool]:
    return quality >= limit