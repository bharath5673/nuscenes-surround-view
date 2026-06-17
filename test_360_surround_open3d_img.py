import os
import cv2
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes

def quaternion_to_rotation_matrix(q):
    """Convert a nuScenes quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,   1 - 2*x**2 - 2*y**2]
    ])

def generate_cylindrical_panoramic_texture(nusc, sample, nuscenes_data_path, pano_shape=(1080, 3840)):
    """Stitches the 6 surround cameras into a high-res flat 2D panoramic canvas."""
    pano_h, pano_w = pano_shape
    y_cols, x_rows = np.meshgrid(np.arange(pano_h), np.arange(pano_w), indexing='ij')
    
    # Map panorama coordinates to cylindrical angles (Theta from -pi to pi)
    theta = (x_rows / pano_w) * 2.0 * np.pi - np.pi  
    focal_length = pano_w / (2.0 * np.pi)  
    h_cyl = (pano_h / 2.0 - y_cols) / focal_length
    
    # --- FIXED: COORDINATE ALIGNMENT ---
    # Aligns CAM_FRONT (0 degrees heading) directly to the center of the panorama.
    X_ego = -np.sin(theta)
    Y_ego = np.cos(theta)
    Z_ego = h_cyl
    pts_ego = np.vstack((X_ego.ravel(), Y_ego.ravel(), Z_ego.ravel()))
    
    cam_channels = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
    accumulated_canvas = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
    weight_canvas = np.zeros((pano_h, pano_w, 1), dtype=np.float32)
    
    for cam_channel in cam_channels:
        data_token = sample['data'][cam_channel]
        cam_data = nusc.get('sample_data', data_token)
        
        img_path = os.path.join(nuscenes_data_path, cam_data['filename'])
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsics = np.array(calib['camera_intrinsic'])
        distortion = calib.get('camera_distortion', None)
        dist_coeffs = np.array(distortion)[:5] if distortion else np.zeros(5)
        
        R_cam_to_ego = quaternion_to_rotation_matrix(calib['rotation'])
        R_ego_to_cam = R_cam_to_ego.T
        pts_cam = R_ego_to_cam @ pts_ego
        
        valid_z = pts_cam[2, :] > 0.1
        map_x = np.full((pano_h, pano_w), -1.0, dtype=np.float32)
        map_y = np.full((pano_h, pano_w), -1.0, dtype=np.float32)
        mask = np.zeros((pano_h, pano_w), dtype=np.uint8)
        
        if np.any(valid_z):
            pts_cam_cv = pts_cam[:, valid_z].T.reshape(-1, 1, 3)
            img_pts, _ = cv2.projectPoints(pts_cam_cv, np.zeros(3), np.zeros(3), intrinsics, dist_coeffs)
            img_pts = img_pts.reshape(-1, 2)
            
            img_w, img_h = img.shape[1], img.shape[0]
            inside_bounds = (
                (img_pts[:, 0] >= 0.5) & (img_pts[:, 0] < img_w - 0.5) &
                (img_pts[:, 1] >= 0.5) & (img_pts[:, 1] < img_h - 0.5)
            )
            
            global_indices = np.where(valid_z)[0][inside_bounds]
            map_x.ravel()[global_indices] = img_pts[inside_bounds, 0]
            map_y.ravel()[global_indices] = img_pts[inside_bounds, 1]
            mask.ravel()[global_indices] = 255
            
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        weight_mask = np.zeros((pano_h, pano_w), dtype=np.float32)
        for y in range(pano_h):
            valid_cols = np.where(mask[y] > 0)[0]
            if len(valid_cols) > 40:
                weight_mask[y, valid_cols] = 1.0
                feather_len = min(30, len(valid_cols) // 2)
                weight_mask[y, valid_cols[:feather_len]] = np.linspace(0, 1, feather_len)
                weight_mask[y, valid_cols[-feather_len:]] = np.linspace(1, 0, feather_len)
                
        weight_mask = np.expand_dims(weight_mask, axis=2)
        accumulated_canvas += warped_img.astype(np.float32) * weight_mask
        weight_canvas += weight_mask
        
    weight_canvas[weight_canvas == 0] = 1.0
    panorama_texture = (accumulated_canvas / weight_canvas).astype(np.uint8)
    return panorama_texture

def create_textured_cylinder(radius=10.0, height=8.0, rows=50, cols=100):
    """Generates a parametric open cylinder mesh mapping 3 explicit UVs per triangle face."""
    mesh = o3d.geometry.TriangleMesh()
    
    # 1. Generate grid vertices
    vertices = []
    for r in range(rows):
        z = (r / (rows - 1)) * height - (height / 2.0)
        for c in range(cols):
            # Angular step adjusted to match the panorama transformation profile
            theta = (c / (cols - 1)) * 2.0 * np.pi - np.pi / 2.0
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, z])
            
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
    
    # 2. Build triangles and their associated explicit UV map coordinates 
    triangles = []
    triangle_uvs = []
    
    for r in range(rows - 1):
        for c in range(cols - 1):
            v0 = r * cols + c
            v1 = v0 + 1
            v2 = (r + 1) * cols + c
            v3 = v2 + 1
            
            u_left  = c / (cols - 1)
            u_right = (c + 1) / (cols - 1)
            
            # --- FIXED: VERTICAL TEXTURE FLIP ---
            # Standard uv grid maps 0 at the bottom. Flipping the index scales the image right-side up.
            v_bottom = 1.0 - (r / (rows - 1))
            v_top    = 1.0 - ((r + 1) / (rows - 1))
            
            # Triangle 1 (Inward-facing configuration)
            triangles.append([v0, v2, v1])
            triangle_uvs.append([u_left,  v_bottom])
            triangle_uvs.append([u_left,  v_top])
            triangle_uvs.append([u_right, v_bottom])
            
            # Triangle 2 (Inward-facing configuration)
            triangles.append([v1, v2, v3])
            triangle_uvs.append([u_right, v_bottom])
            triangle_uvs.append([u_left,  v_top])
            triangle_uvs.append([u_right, v_top])
            
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(triangle_uvs, dtype=np.float64))
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(triangles), dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh

def main():
    nuscenes_data_path = 'mini/v1.0-mini/'
    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_data_path, verbose=False)
    
    # Process the first frame sequence
    sample = nusc.sample[0]
    print("Baking 360 surround-view multi-camera stitched texture map...")
    texture_img = generate_cylindrical_panoramic_texture(nusc, sample, nuscenes_data_path)
    
    print("Constructing 3D theater viewport...")
    cylinder_mesh = create_textured_cylinder(radius=8.0, height=6.0, rows=40, cols=80)
    cylinder_mesh.textures = [o3d.geometry.Image(np.ascontiguousarray(texture_img))]
    
    # --- FIXED: SCALE AND ROTATE EGO CAR BOX ---
    # Slightly downscaled bounding box dimensions for visual proportions
    box_w, box_l, box_h = 1.4, 3.5, 1.3
    ego_box = o3d.geometry.TriangleMesh.create_box(width=box_w, height=box_l, depth=box_h)
    ego_box.compute_vertex_normals()
    ego_box.paint_uniform_color([0.1, 0.8, 0.1])
    
    # Rotate the box exactly 90 degrees counter-clockwise around the global Z-axis
    rotation_matrix = ego_box.get_rotation_matrix_from_xyz(np.array([0, 0, np.radians(90)]))
    ego_box.rotate(rotation_matrix, center=(0, 0, 0))
    
    # Center the rotated box perfectly at the coordinate origin
    ego_box.translate([-box_w/2, -box_l/2, -box_h/2])
    
    print("Launching standard Open3D rendering window canvas...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Seamless 360 Surround Cam Viewer", width=1280, height=720)
    
    vis.add_geometry(cylinder_mesh)
    vis.add_geometry(ego_box)
    
    opt = vis.get_render_option()
    opt.light_on = False
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.05)         # Sit inside cockpit center
    ctr.set_front([0, 1, 0])   # Look straight toward CAM_FRONT
    ctr.set_up([0, 0, 1])      # Coordinate system Z-Up
    ctr.set_lookat([0, 0, 0])
    
    print("\n[SUCCESS] Viewer Operational!")
    print(" -> Left Click + Drag to rotate 360 degrees smoothly inside the bubble.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()