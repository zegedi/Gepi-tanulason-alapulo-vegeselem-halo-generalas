import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_json,
    resolve_geometry_data,
    create_som,
    initialize_weights,
    apply_disabled_weights,
    apply_fixed_weights,
    train_som,
    project_grid_boundary_from_step,
    project_closest_node_on_point,
    quad_mesh_on_step_surface,
    tri_mesh_on_step_surface
)


def main(json_path: str):
    cfg = load_json(json_path)

    cfg = load_json(json_path)

    # 1) Geometry resolution (sampling or direct data)
    geo = cfg["geometry"]
    curve_data, surface_data = resolve_geometry_data(geo)

    print("curve_data\n", curve_data)
    print("surface_data\n", surface_data)

    # Combine training data

    # if isinstance(curve_data, list):
    #     data = np.concatenate((*curve_data, surface_data), axis=0)
    # else:
    #     data = np.concatenate((curve_data, surface_data), axis=0)

    # # Optional visualization
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.show()

    # 2) SOM creation
    som = create_som(cfg)

    # 3) Weight initialization
    initialize_weights(som, cfg)

    # 4) Fixed weights (optional)
    apply_fixed_weights(som, cfg)

    # 5) Disabled weights (optional)
    apply_disabled_weights(som, cfg)

    # 6) Training
    train_som(som, curve_data, surface_data, None, cfg)

    # 7) Export mesh
    weights = som.get_weights()
    msh_file = cfg["output"]["msh_file"]

    # After training
    weights = som.get_weights(
        disable=True,
        disable_value=np.nan
    )

    # Optional boundary projection
    proj_boundary_cfg = cfg.get("postprocess", {}).get("project_boundary", {})

    if proj_boundary_cfg.get("enabled", False):
        print("Projecting weights onto boundary")
        weights = project_grid_boundary_from_step(
            step_path=cfg["geometry"]["step_file"],
            grid=weights,
            z0=proj_boundary_cfg.get("z0", 0.0)
        )

    # Optional node projection
    proj_node_cfg = cfg.get("postprocess", {}).get("project_node", {})

    if proj_node_cfg.get("enabled", False):
        print("Projecting closest weights onto points")
        weights = project_closest_node_on_point(
            step_path=cfg["geometry"]["step_file"],
            grid=weights,
            z0=proj_node_cfg.get("z0", 0.0),
            skip_tags=proj_node_cfg.get("skip_tags", None)
        )

    out = cfg["output"]

    if out.get("mesh_type", "quad") == "tri":
        tri_mesh_on_step_surface(
            weights,
            cfg["geometry"]["step_file"],
            out["msh_file"]
        )
    else:
        quad_mesh_on_step_surface(
            weights,
            cfg["geometry"]["step_file"],
            out["msh_file"],
            out.get("collapse_collinear", False),
            out.get("collinear_tol", 1e-8),
        )

    print(f"Mesh written to: {msh_file}")

    for i in range(weights.shape[0]):
        plt.plot(weights[i, :, 0], weights[i, :, 1], color='blue', linewidth=1)

    # 2. Plot vertical lines (connect points along axis 0 for each index in axis 1)
    for j in range(weights.shape[1]):
        plt.plot(weights[:, j, 0], weights[:, j, 1], color='blue', linewidth=1)

    plt.title("Grid Plot of Array $a$ with Shape $(10, 10, 2)$")
    plt.xlabel("$x$ coordinate")
    plt.ylabel("$y$ coordinate")
    plt.gca().set_aspect('equal')
    # plt.savefig(r'C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\grid_plot_2006.png')


if __name__ == "__main__":
    import sys
    main(sys.argv[1])