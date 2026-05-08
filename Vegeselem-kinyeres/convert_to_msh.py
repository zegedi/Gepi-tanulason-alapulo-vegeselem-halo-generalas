from fileinput import filename
import gmsh
import numpy as np

import gmsh


def abaqus_to_gmsh_with_geometry(inp_path, step_path, msh_path, msh_boundary_path = None):
    nodes = {}
    elements = []

    # ---------------------------
    # parse Abaqus file
    # ---------------------------
    with open(inp_path, "r") as f:
        lines = f.readlines()

    reading_nodes = False
    reading_elements = False

    for line in lines:
        line = line.strip()

        if line.startswith("*Node"):
            reading_nodes = True
            reading_elements = False
            continue

        if line.startswith("*Element"):
            reading_nodes = False
            reading_elements = True
            continue

        if line.startswith("*"):
            reading_nodes = False
            reading_elements = False
            continue

        if reading_nodes and line:
            parts = [x.strip() for x in line.split(",")]
            tag = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            nodes[tag] = (x, y)

        if reading_elements and line:
            parts = [x.strip() for x in line.split(",")]
            conn = list(map(int, parts[1:5]))
            elements.append(conn)

    # ---------------------------
    # build gmsh model
    # ---------------------------
    gmsh.initialize()
    gmsh.open(step_path)

    if msh_boundary_path is not None:
        gmsh.merge(msh_boundary_path)

    # gmsh.model.add("srl_afm_mesh")

    # load geometry
    # gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    # get the single surface
    surfaces = gmsh.model.getEntities(dim=2)
    if len(surfaces) != 1:
        raise RuntimeError("Expected exactly one surface in STEP file")

    surface_tag = surfaces[0][1]

    # ---------------------------
    # add nodes
    # ---------------------------
    node_tags = list(nodes.keys())
    coords = []
    for tag in node_tags:
        x, y = nodes[tag]
        coords.extend([x, y, 0.0])

    gmsh.model.mesh.addNodes(
        dim=2,
        tag=surface_tag,
        nodeTags=node_tags,
        coord=coords
    )

    # ---------------------------
    # add elements
    # ---------------------------
    element_tags = list(range(1, len(elements) + 1))
    flat_conn = [n for elem in elements for n in elem]

    # gmsh quad type = 3
    gmsh.model.mesh.addElementsByType(
        tag=surface_tag,
        elementType=3,
        elementTags=element_tags,
        nodeTags=flat_conn
    )

    gmsh.write(msh_path)
    gmsh.finalize()


if __name__ == "__main__":

    abaqus_to_gmsh_with_geometry(
        r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new\unit_circle.inp",
        r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\test_geo\unit_circle.step",
        r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new\unit_circle.msh"
    )

    # abaqus_to_gmsh_with_geometry(
    #     r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new_smooth\unit_circle.inp",
    #     r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_som\test_geo\unit_circle.step",
    #     r"C:\Users\EGV1BP\Documents\SRL-AssistedAFM-main\SRL-AssistedAFM-main\output\new_smooth\unit_circle.msh"
    # )