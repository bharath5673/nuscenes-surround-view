import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes

def quaternion_to_rotation_matrix(q):
    """Convert a nuScenes quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,   1 - 2*x**2 - 2*y**2]
    ])

def build_cylindrical_maps_with_blending(nusc, sample_data_tokens, pano_shape=(500, 2000)):
    """
    Creates lookup remap meshes for all 6 cameras onto a shared cylindrical canvas.
    Returns the maps, precise label coordinate centers, and individual camera weight masks.
    """
    pano_h, pano_w = pano_shape
    y_cols, x_rows = np.meshgrid(np.arange(pano_h), np.arange(pano_w), indexing='ij')
    
    # Map panorama canvas coordinates to cylindrical angles (Theta from -pi to pi)
    theta = (x_rows / pano_w) * 2.0 * np.pi - np.pi  
    focal_length = pano_w / (2.0 * np.pi)  
    h_cyl = (pano_h / 2.0 - y_cols) / focal_length
    
    # Transform to 3D Unit Vectors in Vehicle Ego Space (X-Right, Y-Front, Z-Up)
    X_ego = np.sin(theta)
    Y_ego = np.cos(theta)
    Z_ego = h_cyl
    pts_ego = np.vstack((X_ego.ravel(), Y_ego.ravel(), Z_ego.ravel())) 
    
    # Storage arrays for each camera's projection mapping
    camera_maps = {}
    camera_centers = {}
    
    for idx, (cam_channel, data_token) in enumerate(sample_data_tokens.items()):
        cam_data = nusc.get('sample_data', data_token)
        calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        intrinsics = np.array(calib['camera_intrinsic'])
        distortion = calib.get('camera_distortion', None)
        dist_coeffs = np.array(distortion)[:5] if distortion else np.zeros(5)
        
        R_cam_to_ego = quaternion_to_rotation_matrix(calib['rotation'])
        R_ego_to_cam = R_cam_to_ego.T
        pts_cam = R_ego_to_cam @ pts_ego 
        
        # Project 3D rays to 2D image coordinates
        valid_z = pts_cam[2, :] > 0.1
        map_x = np.full((pano_h, pano_w), -1.0, dtype=np.float32)
        map_y = np.full((pano_h, pano_w), -1.0, dtype=np.float32)
        mask = np.zeros((pano_h, pano_w), dtype=np.uint8)
        
        if np.any(valid_z):
            pts_cam_cv = pts_cam[:, valid_z].T.reshape(-1, 1, 3)
            img_pts, _ = cv2.projectPoints(pts_cam_cv, np.zeros(3), np.zeros(3), intrinsics, dist_coeffs)
            img_pts = img_pts.reshape(-1, 2)
            
            img_w, img_h = 1600, 900 
            inside_bounds = (
                (img_pts[:, 0] >= 0.5) & (img_pts[:, 0] < img_w - 0.5) &
                (img_pts[:, 1] >= 0.5) & (img_pts[:, 1] < img_h - 0.5)
            )
            
            global_indices = np.where(valid_z)[0][inside_bounds]
            map_x.ravel()[global_indices] = img_pts[inside_bounds, 0]
            map_y.ravel()[global_indices] = img_pts[inside_bounds, 1]
            mask.ravel()[global_indices] = 255
            
        camera_maps[cam_channel] = (map_x, map_y, mask)
        
        # Calculate geometric center of the camera in panorama coordinate space for proper labels
        cam_forward_vector = R_cam_to_ego @ np.array([0, 0, 1]) # Camera optical axis
        cam_yaw = np.arctan2(cam_forward_vector[0], cam_forward_vector[1]) # Angle in Ego Space
        center_x_px = int(((cam_yaw + np.pi) / (2.0 * np.pi)) * pano_w)
        camera_centers[cam_channel] = center_x_px

    return camera_maps, camera_centers

# --- Main Pipeline Setup ---
nuscenes_data_path = 'mini/v1.0-mini/'
nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_data_path, verbose=False)

# Seamless 360 layout order
cam_channels = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
pano_shape = (500, 2000) 
video_writer = cv2.VideoWriter("nuscenes_360_surroundcam_panorama.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 4, (pano_shape[1], pano_shape[0]))

print("Generating seamless calibrated panoramic stream...")

for sample in nusc.sample:
    sample_tokens = {cam: sample['data'][cam] for cam in cam_channels}
    
    # 1. Compute maps, pixel masks, and geometric text centers
    camera_maps, camera_centers = build_cylindrical_maps_with_blending(nusc, sample_tokens, pano_shape=pano_shape)
    
    # Track accumulated images and dynamic weights for blending normalization
    accumulated_canvas = np.zeros((pano_shape[0], pano_shape[1], 3), dtype=np.float32)
    weight_canvas = np.zeros((pano_shape[0], pano_shape[1], 1), dtype=np.float32)
    
    # 2. Warp images and generate soft alpha blending weights at boundaries
    for cam_channel in cam_channels:
        cam_data = nusc.get('sample_data', sample_tokens[cam_channel])
        img_path = os.path.join(nuscenes_data_path, cam_data['filename'])
        img = cv2.imread(img_path)
        if img is None: continue
            
        map_x, map_y, mask = camera_maps[cam_channel]
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Build a feathering weight mask to fade edges end-to-end cleanly
        weight_mask = np.zeros((pano_shape[0], pano_shape[1]), dtype=np.float32)
        for y in range(pano_shape[0]):
            valid_cols = np.where(mask[y] > 0)[0]
            if len(valid_cols) > 40:
                # Keep center strong (1.0), soften endpoints (0.0 to 1.0) over a 20px window
                weight_mask[y, valid_cols] = 1.0
                feather_len = min(20, len(valid_cols) // 2)
                
                # Fade-in left edge, Fade-out right edge
                weight_mask[y, valid_cols[:feather_len]] = np.linspace(0, 1, feather_len)
                weight_mask[y, valid_cols[-feather_len:]] = np.linspace(1, 0, feather_len)
                
        weight_mask = np.expand_dims(weight_mask, axis=2)
        accumulated_canvas += warped_img.astype(np.float32) * weight_mask
        weight_canvas += weight_mask
        
    # 3. Normalize blending math to prevent dark artifact seams
    weight_canvas[weight_canvas == 0] = 1.0
    panorama_canvas = (accumulated_canvas / weight_canvas).astype(np.uint8)
        
    # 4. Burn in labels at their exact geometric centers
    for cam_name, center_x in camera_centers.items():
        # Shift slightly left to center-align the string length
        text_x = (center_x - 50) % pano_shape[1]
        
        # Draw a clean drop-shadow label look
        cv2.putText(panorama_canvas, cam_name, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(panorama_canvas, cam_name, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Calibrated 360 Cylindrical Panorama', panorama_canvas)
    video_writer.write(panorama_canvas)
    
    if cv2.waitKey(1) == 27: # Press ESC to break
        break

video_writer.release()
cv2.destroyAllWindows()
print("Execution Complete! Output cleanly written to calibrated_360_panorama.mp4")