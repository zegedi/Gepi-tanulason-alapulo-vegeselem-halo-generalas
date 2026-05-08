import gmsh
import numpy as np
import pandas as pd

from metric import (
    jacobian_ratio_error,
    jacobian_ratio_warning,
    max_corner_angle,
    parallel_deviation_warning,
    parallel_deviation_error,
    triangle_aspect_ratio,
    triangle_jacobian_ratio,
    triangle_max_corner_angle_error,
    triangle_max_corner_angle_warning,
    triangle_parallel_deviation,
    aspect_ratio_error,
    aspect_ratio_warnings,
    quad_max_corner_angle_error,
    quad_max_corner_angle_warning,
    quad_aspect_ratio,
    quad_jacobian_ratio,
    quad_max_corner_angle_error,
    quad_max_corner_angle_warning,
    quad_parallel_deviation
)



def main(mesh: str, quality: str, summary: str, reverse: bool = False) -> None:

    gmsh.initialize()
    gmsh.open(mesh)

    XY_INDEX = slice(2)
    QUAD_SHAPE = (-1, 4)
    COORD_SHAPE = (-1, 3)
    TRIANGLE_SHAPE = (-1, 3)

    # Get all node informations from the mesh.
    nodes, coords, _ = gmsh.model.mesh.getNodes()
    
    # Get the X and Y coordinates of the nodes.
    coords = np.reshape(coords, COORD_SHAPE)[..., XY_INDEX]

    # Get all node element informations from the mesh.
    TRIANGLE_TYPE = 2
    QUAD_TYPE = 3
    ZERO_BASED_INDEX = 1

    triangles, triangle_nodes = gmsh.model.mesh.getElementsByType(TRIANGLE_TYPE)
    quads, quad_nodes = gmsh.model.mesh.getElementsByType(QUAD_TYPE)

    triangle_nodes -= ZERO_BASED_INDEX
    quad_nodes -= ZERO_BASED_INDEX

    quad_nodes = np.reshape(quad_nodes, QUAD_SHAPE)
    triangle_nodes = np.reshape(triangle_nodes, TRIANGLE_SHAPE)

    if reverse:
        triangle_nodes = np.fliplr(triangle_nodes)
        quad_nodes = np.fliplr(quad_nodes)

    # Aspect ratio metrics.
    tri_aspect_ratio = triangle_aspect_ratio(coords, triangle_nodes)
    qua_aspect_ratio = quad_aspect_ratio(coords, quad_nodes)

    # Jacobian ratio metrics.
    tri_jacobian_ratio = triangle_jacobian_ratio(coords, triangle_nodes)
    qua_jacobian_ratio = quad_jacobian_ratio(coords, quad_nodes)
    
    # Max corner angle metrics.
    tri_max_corner_angle = max_corner_angle(coords, triangle_nodes)
    quad_max_corner_angle = max_corner_angle(coords, quad_nodes)

    # Parallel deviation metrics.
    tri_parallel_deviation = triangle_parallel_deviation(coords, triangle_nodes)
    qua_parallel_deviation = quad_parallel_deviation(coords, quad_nodes)

    # Gmsh quality metrics.
    GAMMA = "gamma"
    SICN = "minSICN"
    SIGE = "minSIGE"
    tri_gamma = gmsh.model.mesh.getElementQualities(triangles, GAMMA)
    qua_gamma = gmsh.model.mesh.getElementQualities(quads, GAMMA)
    tri_sicn = gmsh.model.mesh.getElementQualities(triangles, SICN)
    qua_sicn = gmsh.model.mesh.getElementQualities(quads, SICN)
    tri_sige = gmsh.model.mesh.getElementQualities(triangles, SIGE)
    qua_sige = gmsh.model.mesh.getElementQualities(quads, SIGE)


    #
    quality_dict = {
        "Tag": np.concatenate((triangles, quads)),
        "Aspect Ratio": np.concatenate((tri_aspect_ratio, qua_aspect_ratio)),
        "Jacobian Ratio": np.concatenate((tri_jacobian_ratio, qua_jacobian_ratio)),
        "Parallel Deviation": np.concatenate((tri_parallel_deviation, qua_parallel_deviation)),
        "Maximum Corner Angle": np.concatenate((tri_max_corner_angle, quad_max_corner_angle)),
        "Gamma":  np.concatenate((tri_gamma, qua_gamma)),
        "Signed Inverted Condition Number":  np.concatenate((tri_sicn, qua_sicn)),
        "Signed Inverted Gradient Error":  np.concatenate((tri_sige, qua_sige))
    }

    quality_df = pd.DataFrame(quality_dict)
    quality_df.to_csv(quality, index=False)

    # Parallel deviation limits.
    parallel_deviation_warn = \
        parallel_deviation_warning(quality_dict["Parallel Deviation"])
    parallel_deviation_err = \
        parallel_deviation_error(quality_dict["Parallel Deviation"])
    parallel_deviation_warn_err = \
        parallel_deviation_warn | parallel_deviation_err
    
    # Aspect ratio limits.
    aspect_ratio_warn = aspect_ratio_warnings(quality_dict["Aspect Ratio"])
    aspect_ratio_err = aspect_ratio_error(quality_dict["Aspect Ratio"])
    aspect_ratio_warn_err = aspect_ratio_warn | aspect_ratio_err

    # Jacobian ratio limits.
    jacobian_ratio_warn = jacobian_ratio_warning(quality_dict["Jacobian Ratio"])
    jacobian_ratio_err = jacobian_ratio_error(quality_dict["Jacobian Ratio"])
    jacobian_ratio_warn_err = jacobian_ratio_warn | jacobian_ratio_err

    # Max corner angle limits.
    max_corner_angle_warn = np.concatenate((
        triangle_max_corner_angle_warning(tri_max_corner_angle),
        quad_max_corner_angle_warning(quad_max_corner_angle)
    ))
    max_corner_angle_err = np.concatenate((
        triangle_max_corner_angle_error(tri_max_corner_angle),
        quad_max_corner_angle_error(quad_max_corner_angle)
    ))
    max_corner_angle_warn_err = max_corner_angle_warn | max_corner_angle_err

    # Any
    any_warn = \
        parallel_deviation_warn | aspect_ratio_warn | \
        jacobian_ratio_warn | max_corner_angle_warn
    any_err  = \
        parallel_deviation_err | aspect_ratio_err | \
        jacobian_ratio_err | max_corner_angle_err
    any_warn_err = any_warn | any_err


    summary_dict = {
        "Number tested": [
            len(quality_dict["Aspect Ratio"]),
            len(quality_dict["Jacobian Ratio"]),
            len(quality_dict["Parallel Deviation"]),
            len(quality_dict["Maximum Corner Angle"]),
            len(any_warn_err),
            sum((
                len(quality_dict["Aspect Ratio"]),
                len(quality_dict["Jacobian Ratio"]),
                len(quality_dict["Parallel Deviation"]),
                len(quality_dict["Maximum Corner Angle"]),
            ))
        ],
        "Warning count": [
            np.count_nonzero(aspect_ratio_warn),
            np.count_nonzero(jacobian_ratio_warn),
            np.count_nonzero(parallel_deviation_warn),
            np.count_nonzero(max_corner_angle_warn),
            np.count_nonzero(any_warn),
            np.sum([
                np.count_nonzero(aspect_ratio_warn),
                np.count_nonzero(jacobian_ratio_warn),
                np.count_nonzero(parallel_deviation_warn),
                np.count_nonzero(max_corner_angle_warn)
            ])
        ],
        "Error count": [
            np.count_nonzero(aspect_ratio_err),
            np.count_nonzero(jacobian_ratio_err),
            np.count_nonzero(parallel_deviation_err),
            np.count_nonzero(max_corner_angle_err),
            np.count_nonzero(any_err),
            np.sum([
                np.count_nonzero(aspect_ratio_err),
                np.count_nonzero(jacobian_ratio_err),
                np.count_nonzero(parallel_deviation_err),
                np.count_nonzero(max_corner_angle_err)
            ])
        ],
        "Warn+Err %": [
            np.count_nonzero(aspect_ratio_warn_err) / len(aspect_ratio_warn_err),
            np.count_nonzero(jacobian_ratio_warn_err) / len(jacobian_ratio_warn_err),
            np.count_nonzero(parallel_deviation_warn_err) / len(parallel_deviation_warn_err),
            np.count_nonzero(max_corner_angle_warn_err) / len(max_corner_angle_warn_err),
            np.count_nonzero(any_warn_err) / len(any_warn_err),
            sum((
                np.count_nonzero(aspect_ratio_warn_err),
                np.count_nonzero(jacobian_ratio_warn_err),
                np.count_nonzero(parallel_deviation_warn_err),
                np.count_nonzero(max_corner_angle_warn_err)
            )) / 
            sum((
                len(aspect_ratio_warn_err),
                len(jacobian_ratio_warn_err),
                len(parallel_deviation_warn_err),
                len(max_corner_angle_warn_err)
            ))
        ],
    }

    summary_df = pd.DataFrame(
        summary_dict, 
        index=[
            "Aspect Ratio", 
            "Jacobian Ratio", 
            "Parallel Deviation", 
            "Maximum Corner Angle",
            "Any",
            "All"
        ]
    )
    summary_df.to_csv(summary, index=True)

    gmsh.finalize()


if __name__ == "__main__":
    # main(
    #     r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new\unit_circle.msh",
    #     r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new\unit_circle_quality.csv",
    #     r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new\unit_circle_summary.csv"
    # )
    # main(
    #     r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new_smooth\full_5_1200.msh",
    #     r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new_smooth\full_5_1200_quality.csv",
    #     r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new_smooth\full_5_1200_summary.csv",
    #     reverse=False
    # )
    # main(
    #     r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\test_geo\gmsh\lshape.msh",
    #     r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\test_geo\gmsh\lshape_quality.csv",
    #     r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\test_geo\gmsh\lshape_summary.csv",
    #     reverse=False
    # )
    # main(
    #     r"C:\Users\EGV1BP\Documents\FEM+AI\gmsh-project\parts\full\full_5_500_mesh.msh",
    #     r"C:\Users\EGV1BP\Documents\FEM+AI\gmsh-project\parts\full\full_5_500_mesh_quality.csv",
    #     r"C:\Users\EGV1BP\Documents\FEM+AI\gmsh-project\parts\full\full_5_500_mesh_summary.csv",
    #     reverse=False
    # )
    main(
        r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\test_geo\2006\unit_circle.msh",
        r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\test_geo\2006\unit_circle_quality.csv",
        r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\test_geo\2006\unit_circle_summary.csv",
        reverse=True
    )