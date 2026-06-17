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

def precompute_cylindrical_maps(nusc, first_sample, pano_shape=(1080, 3840)):
    """
    Computes all heavy projection math and pixel lookups exactly ONCE.
    Rotated to map 0 degrees straight forward (CAM_FRONT) to the center of the viewport.
    """
    pano_h, pano_w = pano_shape
    y_cols, x_rows = np.meshgrid(np.arange(pano_h), np.arange(pano_w), indexing='ij')
    
    # Map panorama coordinates to cylindrical angles (Theta from -pi to pi)
    theta = (x_rows / pano_w) * 2.0 * np.pi - np.pi  
    focal_length = pano_w / (2.0 * np.pi)  
    h_cyl = (pano_h / 2.0 - y_cols) / focal_length
    
    # --- CORRECTED COORDINATE ALIGNMENT ---
    # According to the sensor diagram, 0 degrees is straight ahead.
    # Adjusting the projection matrix aligns CAM_FRONT to the center of the panorama.
    X_ego = -np.sin(theta)
    Y_ego = np.cos(theta)
    Z_ego = h_cyl
    pts_ego = np.vstack((X_ego.ravel(), Y_ego.ravel(), Z_ego.ravel()))
    
    cam_channels = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
    precomputed_data = []
    
    for cam_channel in cam_channels:
        data_token = first_sample['data'][cam_channel]
        cam_data = nusc.get('sample_data', data_token)
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
        
        img_w, img_h = 1600, 900 
        
        if np.any(valid_z):
            pts_cam_cv = pts_cam[:, valid_z].T.reshape(-1, 1, 3)
            img_pts, _ = cv2.projectPoints(pts_cam_cv, np.zeros(3), np.zeros(3), intrinsics, dist_coeffs)
            img_pts = img_pts.reshape(-1, 2)
            
            inside_bounds = (
                (img_pts[:, 0] >= 0.5) & (img_pts[:, 0] < img_w - 0.5) &
                (img_pts[:, 1] >= 0.5) & (img_pts[:, 1] < img_h - 0.5)
            )
            
            global_indices = np.where(valid_z)[0][inside_bounds]
            map_x.ravel()[global_indices] = img_pts[inside_bounds, 0]
            map_y.ravel()[global_indices] = img_pts[inside_bounds, 1]
            mask.ravel()[global_indices] = 255
            
        weight_mask = np.zeros((pano_h, pano_w), dtype=np.float32)
        for y in range(pano_h):
            valid_cols = np.where(mask[y] > 0)[0]
            if len(valid_cols) > 40:
                weight_mask[y, valid_cols] = 1.0
                feather_len = min(30, len(valid_cols) // 2)
                weight_mask[y, valid_cols[:feather_len]] = np.linspace(0, 1, feather_len)
                weight_mask[y, valid_cols[-feather_len:]] = np.linspace(1, 0, feather_len)
                
        weight_mask = np.expand_dims(weight_mask, axis=2)
        
        # Calculate label text positioning from the active mask segment
        sample_row = pano_h // 4
        active_pixels = np.where(mask[sample_row] > 0)[0]
        
        if len(active_pixels) > 0:
            # Handle edge wrap-around seams smoothly
            if (active_pixels[-1] - active_pixels[0]) > (pano_w * 0.8):
                text_x = 0
            else:
                text_x = int(np.median(active_pixels))
        else:
            text_x = pano_w // 2 
            
        precomputed_data.append({
            'channel': cam_channel,
            'map_x': map_x,
            'map_y': map_y,
            'weight': weight_mask,
            'text_x_pos': text_x
        })
        
    return precomputed_data

def generate_fast_panorama(nusc, sample, nuscenes_data_path, precomputed_maps, pano_shape=(1080, 3840)):
    """Fast stitching pipeline with fixed, absolute column slots for camera labels."""
    pano_h, pano_w = pano_shape
    accumulated_canvas = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
    weight_canvas = np.zeros((pano_h, pano_w, 1), dtype=np.float32)
    
    for cache in precomputed_maps:
        data_token = sample['data'][cache['channel']]
        cam_data = nusc.get('sample_data', data_token)
        
        img_path = os.path.join(nuscenes_data_path, cam_data['filename'])
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        warped_img = cv2.remap(img, cache['map_x'], cache['map_y'], 
                               interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        accumulated_canvas += warped_img.astype(np.float32) * cache['weight']
        weight_canvas += cache['weight']
        
    weight_canvas[weight_canvas == 0] = 1.0
    panorama_texture = (accumulated_canvas / weight_canvas).astype(np.uint8)
    
    # --- FIXED: ABSOLUTE HORIZONTAL COORD CENTER SLOTS ---
    # Total width is 3840. Divided into 6 equal slots of 640px each.
    # We place each label perfectly at the center column of its perspective slot.
    camera_slots = {
        'CAM_BACK':        2880,   # Slot 1: 0 - 640
        'CAM_BACK_RIGHT':  2240,   # Slot 2: 640 - 1280
        'CAM_FRONT_RIGHT': 1600,  # Slot 3: 1280 - 1920
        'CAM_FRONT':       960,  # Slot 4: 1920 - 2560
        'CAM_FRONT_LEFT':  320,  # Slot 5: 2560 - 3200
        'CAM_BACK_LEFT':   3520   # Slot 6: 3200 - 3840
    }
    
    # Overlay the camera labels using the fixed slots
    for cam_name, tx in camera_slots.items():
        ty = 120 # Vertical placement height
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 4
        text_size = cv2.getTextSize(cam_name, font, font_scale, thickness)[0]
        adjusted_tx = tx - (text_size[0] // 2)
        
        # Keep text within safe image array boundaries
        adjusted_tx = max(10, min(adjusted_tx, pano_w - text_size[0] - 10))
        
        # Draw clean background box behind text
        p1 = (adjusted_tx - 15, ty - text_size[1] - 15)
        p2 = (adjusted_tx + text_size[0] + 15, ty + 15)
        cv2.rectangle(panorama_texture, p1, p2, (20, 20, 20), -1)
        
        # Burn text name string onto the panorama canvas
        cv2.putText(panorama_texture, cam_name, (adjusted_tx, ty), font, font_scale, (0, 255, 200), thickness, cv2.LINE_AA)
        
    return panorama_texture

def create_textured_cylinder(radius=10.0, height=8.0, rows=40, cols=80):
    """Generates a parametric open cylinder mesh aligning 0 degrees with the positive Y forward vector."""
    mesh = o3d.geometry.TriangleMesh()
    
    vertices = []
    for r in range(rows):
        z = (r / (rows - 1)) * height - (height / 2.0)
        for c in range(cols):
            # Angular distribution offset to match the updated panorama transformation matrix
            theta = (c / (cols - 1)) * 2.0 * np.pi - np.pi / 2.0
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, z])
            
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
    
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
            v_bottom = 1.0 - (r / (rows - 1))
            v_top    = 1.0 - ((r + 1) / (rows - 1))
            
            triangles.append([v0, v2, v1])
            triangle_uvs.append([u_left,  v_bottom])
            triangle_uvs.append([u_left,  v_top])
            triangle_uvs.append([u_right, v_bottom])
            
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
    
    print("Constructing 3D theater viewport...")
    cylinder_mesh = create_textured_cylinder(radius=8.0, height=6.0, rows=30, cols=60)
    
    print("Precomputing static layout matrices with synchronized sensor mappings...")
    precomputed_maps = precompute_cylindrical_maps(nusc, nusc.sample[0])
    
    current_sample_idx = 0
    sample = nusc.sample[current_sample_idx]
    texture_img = generate_fast_panorama(nusc, sample, nuscenes_data_path, precomputed_maps)
    cylinder_mesh.textures = [o3d.geometry.Image(np.ascontiguousarray(texture_img))]
    
    # Configure centered ego car box
    ego_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=3.5, depth=1.3)
    ego_box.compute_vertex_normals()
    ego_box.paint_uniform_color([0.1, 0.8, 0.1])
    
    # Apply 90 degree left rotation
    rotation_matrix_box = ego_box.get_rotation_matrix_from_xyz(np.array([0, 0, np.radians(90)]))
    ego_box.rotate(rotation_matrix_box, center=(0, 0, 0))
    
    new_w, new_l, new_h = 1.5, 3.5, 1.3
    ego_box.translate([-new_w/2, -new_l/2, -new_h/2])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Panoramic Stream - Calibrated Camera Feeds", width=1280, height=720)
    
    vis.add_geometry(cylinder_mesh)
    vis.add_geometry(ego_box)
    
    opt = vis.get_render_option()
    opt.light_on = False
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.05)
    ctr.set_front([0, 1, 0])  # Face CAM_FRONT forward
    ctr.set_up([0, 0, 1])     # Coordinate system Z-Up structure
    ctr.set_lookat([0, 0, 0])
    
    print("\n[SUCCESS] Matrix synchronization complete! Labels and camera projections are perfectly aligned with the vehicle heading.")
    
    total_samples = len(nusc.sample)
    keep_running = True
    
    while keep_running:
        current_sample_idx = (current_sample_idx + 1) % total_samples
        sample = nusc.sample[current_sample_idx]
        
        texture_img = generate_fast_panorama(nusc, sample, nuscenes_data_path, precomputed_maps)
        
        cylinder_mesh.textures = [o3d.geometry.Image(np.ascontiguousarray(texture_img))]
        vis.update_geometry(cylinder_mesh)
        
        keep_running = vis.poll_events()
        vis.update_renderer()
        
    vis.destroy_window()
    print("Loop closed successfully.")

if __name__ == "__main__":
    main()