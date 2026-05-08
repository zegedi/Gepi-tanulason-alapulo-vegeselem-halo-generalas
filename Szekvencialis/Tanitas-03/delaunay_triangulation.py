import gmsh
import sys
import os
import numpy as np
import triangle as tr
from shapely import MultiPolygon, Polygon
from scipy.spatial import cKDTree, Delaunay
from alphashape import alphashape

def create_mesh_from_result_txt(geometry_file_path, result_file_path):

    # 1. Read in the geometry file (as specified)
    print(f"Opening geometry file: {geometry_file_path}")
    gmsh.open(geometry_file_path)

    surfaces = gmsh.model.getEntities(2)
    
    # Store surface_tag and adjust the default rectangle creation
    surface_tag = -1 # Initialize with an invalid tag
    
    # 2. Collect all vertex coordinates from result.txt
    print(f"Processing result file to collect vertices: {result_file_path}")
    all_obj_points = [] # List to store all [x, y, z] coordinates from OBJ
    
    # Placeholder for min/max coordinates to define default surface later
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    with open(result_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    all_obj_points.append([x, y, z])
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                except ValueError:
                    print(f"Warning: Malformed vertex data on line: {line.strip()}. Skipping.")

    if not all_obj_points:
        raise RuntimeError("No valid vertex 'v' lines found in result.txt. Cannot proceed with triangulation.")

    points_array = np.array(all_obj_points) # Convert to numpy array for easier indexing

    # Define a buffer around the bounding box for the default surface
    buffer = 1.0 # 1 unit buffer
    rect_x_min = min_x - buffer
    rect_y_min = min_y - buffer
    rect_width = (max_x - min_x) + 2 * buffer
    rect_height = (max_y - min_y) + 2 * buffer

    if not surfaces:
        print("No surfaces found in the geometry file. Creating a default rectangle surface based on point bounding box.")
        gmsh.model.add("temp_model_for_mesh")
        # For simplicity, let's assume the Z-level for the default surface is 0.
        gmsh.model.occ.addRectangle(rect_x_min, rect_y_min, 0, rect_width, rect_height)
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError("Failed to create a surface for mesh attachment.")
        surface_tag = surfaces[0][1] # Get the tag of the first surface
        print(f"Created default surface with tag: {surface_tag} (bbox: {rect_x_min},{rect_y_min} to {rect_x_min+rect_width},{rect_y_min+rect_height})")
    else:
        surface_tag = surfaces[0][1]
        print(f"Using existing surface with tag: {surface_tag} from {geometry_file_path}")

    surface_dimension = 2

    # 3. Perform Delaunay triangulation using scipy.spatial.Delaunay on *all* points
    print(f"Performing Delaunay triangulation on {len(points_array)} points...")
    
    unique_z = np.unique(points_array[:, 2])
    if len(unique_z) > 1:
        print("Warning: Z-coordinates are not all the same. Delaunay triangulation will be performed on X,Y plane.")
        print("This might result in a mesh that doesn't accurately represent 3D geometry.")
        print("Consider projecting points to a plane or using a 3D meshing approach if needed.")


    # alpha = optimizealpha(points_array[:, :2], upper=0.5)
    boundary = alphashape(points_array[:, :2], alpha=0.2)

    if not isinstance(boundary, Polygon):
        boundary = alphashape(points_array[:, :2], alpha=0.0)

    exterior_coords = np.array(boundary.exterior.coords)

    # Map back to original point indices efficiently
    tree = cKDTree(points_array[:, :2])
    _, indices = tree.query(exterior_coords[:-1])  # exclude duplicate last vertex

    # Build edge list, closing the loop
    segments = np.column_stack([indices, np.roll(indices, -1)])

    # print("Edge indices (closed polygon):")
    # print(segments)

    # Map back to original indices
    # indices = []
    # for xy in exterior_coords[:-1]:
    #     idx = np.where(np.all(np.isclose(points_array[:, :2], xy, atol=1e-8), axis=1))[0][0]
    #     indices.append(idx)
    # indices = np.array(indices)

    # # Build edge list
    # segments = np.column_stack([indices, np.roll(indices, -1)])[:-1]

    # print("Edge indices:")
    # print(segments)


    data = dict(
        vertices=points_array[:, :2],
        segments=segments 
    )

    tri = tr.triangulate(data, opts='p') # Use X, Y for 2D Delaunay
    # tri = Delaunay(points_array[:, :2])

    all_triangles = tri['triangles'] # These are 0-based indices into points_array
    # all_triangles = tri.simplices

    print(f"Initial Delaunay triangulation generated {len(all_triangles)} triangles.")


    # 4. Add ALL nodes to Gmsh first
    # Create explicit Gmsh node tags (1-based)
    gmsh_node_tags = list(range(1, len(points_array) + 1))
    gmsh_node_coords = points_array.flatten() # Flatten the (N,3) array to (N*3)
    
    gmsh.model.mesh.addNodes(surface_dimension, surface_tag, gmsh_node_tags, gmsh_node_coords)
    print(f"Added {len(gmsh_node_tags)} nodes to Gmsh.")

    # 5. Add ALL Delaunay triangles to Gmsh
    element_type = 2 # 3-node triangle
    
    gmsh_element_tags = []
    gmsh_element_node_tags_flat = []
    
    current_element_tag = 0
    for simplex_indices_0based in all_triangles:
        current_element_tag += 1
        gmsh_element_tags.append(current_element_tag)
        
        # Convert original 0-based indices to Gmsh 1-based tags
        # (which match our assigned gmsh_node_tags)
        gmsh_node_ids_for_element = [idx + 1 for idx in simplex_indices_0based]
        gmsh_element_node_tags_flat.extend(gmsh_node_ids_for_element)

    if gmsh_element_tags:
        gmsh.model.mesh.addElements(
            surface_dimension, surface_tag, [element_type], 
            [gmsh_element_tags], [gmsh_element_node_tags_flat]
        )
        print(f"Added {len(gmsh_element_tags)} initial triangular elements to Gmsh.")
    else:
        print("No triangles generated by Delaunay triangulation to add.")
        gmsh.finalize() # Nothing to mesh, exit early
        return

    # Now, identify and remove triangles whose centroids are outside the surface
    print(f"Checking {len(gmsh_element_tags)} elements for centroids outside surface {surface_tag}...")
    elements_to_delete = []

    # Get all 2D elements associated with our surface
    # gmsh.model.mesh.getElements(dimension, tag) returns (elementTypes, elementTags, nodeTags)
    
    # Get nodes first to access their coordinates
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(surface_dimension, surface_tag)
    # Convert node_coords into a dictionary for quick lookup: {tag: (x, y, z)}
    node_coord_map = {node_tags[i]: (node_coords[i*3], node_coords[i*3+1], node_coords[i*3+2]) 
                      for i in range(len(node_tags))}

    element_types_read, element_tags_read, node_tags_per_element_read = gmsh.model.mesh.getElements(surface_dimension, surface_tag)
    
    # We are interested in the first element type (triangle, type 2)
    if element_types_read and element_types_read[0] == 2: # Ensure it's a triangle type
        triangle_tags = element_tags_read[0]
        triangle_node_tags_flat = node_tags_per_element_read[0]
        
        num_triangles = len(triangle_tags)
        
        for i in range(num_triangles):
            tag = triangle_tags[i]
            # Node tags for the current triangle (3 nodes)
            n1_tag, n2_tag, n3_tag = triangle_node_tags_flat[i*3], triangle_node_tags_flat[i*3+1], triangle_node_tags_flat[i*3+2]
            
            # Get coordinates using the map
            p1 = node_coord_map[n1_tag]
            p2 = node_coord_map[n2_tag]
            p3 = node_coord_map[n3_tag]
            
            # Calculate centroid
            centroid_x = (p1[0] + p2[0] + p3[0]) / 3.0
            centroid_y = (p1[1] + p2[1] + p3[1]) / 3.0
            centroid_z = (p1[2] + p2[2] + p3[2]) / 3.0
            
            is_centroid_inside = gmsh.model.isInside(
                surface_dimension, surface_tag,
                [centroid_x, centroid_y, centroid_z]
            )
            
            if not is_centroid_inside:
                elements_to_delete.append(tag)
    else:
        print("No triangular elements found to filter.")

    if elements_to_delete:
        print(f"Deleting {len(elements_to_delete)} elements whose centroids are outside the surface.")
        # gmsh.model.mesh.removeElements expects (dim, tag, elementTags)
        # elementTags are elementTags. The first elementTypes_read[0] is the elementType (2 for triangles).
        # We need to specify the correct element type for deletion.
        # This becomes a bit tricky if there are multiple element types, but for now we expect only triangles.
        
        # Note: Gmsh's removeElements takes lists for elementTypes and elementTags,
        # but for a single operation, it's typically [elementType], [elementTags...].
        gmsh.model.mesh.removeElements(
            surface_dimension, surface_tag, elements_to_delete
        )
        print(f"Remaining elements: {len(gmsh_element_tags) - len(elements_to_delete)}")
    else:
        print("All triangle centroids are inside the surface. No elements deleted.")

    # After deleting elements, some nodes might become isolated (not connected to any element).
    # Gmsh provides a way to remove these.

    # Determine the output filename
    # os.path.splitext separates the filename into (root, ext)
    result_name, _ = os.path.splitext(os.path.basename(result_file_path))
    output_filename = \
        r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_model_v8_hole_in_square\test_vertex_delaunay_triang" + f"\\{result_name}.msh"

    print(f"Output will be written to '{output_filename}'...")
    gmsh.write(output_filename)

    image_filename = \
        r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_model_v8_hole_in_square\test_vertex_delaunay_triang_png" + f"\\{result_name}.png"

    print(f"Output will be written to '{image_filename}'...")
    gmsh.write(image_filename)

    # if '-nopopup' not in sys.argv:
    #    gmsh.fltk.run()

    # gmsh.finalize()



if __name__ == '__main__':

    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 1) # Show Gmsh messages in terminal

    gmsh.fltk.initialize()

    geo_start = 1
    obj_range = range(1, 2001)

    geo_path_and_name = r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_dataset_version_2.4\Test\Geometry (STEP)\sample_"
    obj_path_and_name = r"C:\Users\EGV1BP\Documents\FEM+AI\fem_ai_model_v8_hole_in_square\tests_vertex\result_"
    
    for geo_index, obj_index in enumerate(obj_range, geo_start):

        geo_path = f"{geo_path_and_name}{geo_index}.step"
        obj_path = f"{obj_path_and_name}{obj_index}.obj"

        create_mesh_from_result_txt(geo_path, obj_path)

    gmsh.fltk.finalize()
    gmsh.finalize()
    
    # create_mesh_from_result_txt("tests/sample_1.step", "tests/result_1.obj")
    # create_mesh_from_result_txt("tests/sample_2.step", "tests/result_2.obj")