# NuScenes 3D Surround View Visualization

A lightweight and educational implementation of a 360° surround-view visualization using the NuScenes API.

The goal of this project is to demonstrate how multi-camera data can be transformed into a common vehicle-centric coordinate system and visualized in a unified top-down view. The implementation prioritizes simplicity, readability, and learning over production-level complexity.

---

## Overview

Modern autonomous driving systems rely on multiple cameras placed around the vehicle to perceive the surrounding environment. Each camera provides a different perspective, making it necessary to transform all observations into a common reference frame.

This project demonstrates:

- Loading synchronized camera data from NuScenes
- Using camera calibration parameters provided by NuScenes
- Transforming points between coordinate systems
- Visualizing camera locations and orientations
- Building a simple 360° surround-view representation
- Rendering 3D bounding boxes in a common coordinate frame

---

## Objectives

The project is designed to help understand:

1. NuScenes dataset structure
2. Camera intrinsics and extrinsics
3. Ego vehicle coordinate system
4. Sensor-to-ego transformations
5. Ego-to-global transformations
6. Multi-camera spatial alignment
7. Bird's-Eye View (BEV) visualization
8. 3D bounding box projection

---

## Dataset

This implementation uses the NuScenes dataset and the official NuScenes API.

Available camera streams:

- CAM_FRONT
- CAM_FRONT_LEFT
- CAM_FRONT_RIGHT
- CAM_BACK
- CAM_BACK_LEFT
- CAM_BACK_RIGHT

Together, these cameras provide full 360° coverage around the vehicle.

---

## Coordinate Systems

NuScenes uses multiple coordinate frames:

### Global Frame

A fixed world coordinate system used across the entire scene.

### Ego Vehicle Frame

A vehicle-centric coordinate system whose origin is located at the ego vehicle.

### Sensor Frame

A coordinate system local to each camera or sensor.

### Image Frame

The 2D pixel coordinate system of the camera image.

---

## Transformation Pipeline

The visualization pipeline follows the transformation chain below:

```text
Global Coordinates
        ↓
Ego Vehicle Coordinates
        ↓
Camera Coordinates
        ↓
Image Coordinates
```

For surround-view generation, the reverse transformation is often used:

```text
Image Coordinates
        ↓
Camera Coordinates
        ↓
Ego Vehicle Coordinates
        ↓
Bird's-Eye View
```

---

## Methodology

### Step 1: Load NuScenes Sample

A sample frame is loaded using the NuScenes API.

### Step 2: Retrieve Camera Calibration

For each camera:

- Intrinsic matrix
- Translation
- Rotation

are obtained from the calibrated sensor records.

### Step 3: Transform to Ego Frame

Camera poses are transformed into the ego vehicle coordinate frame.

### Step 4: Generate Surround View

All camera positions are projected into a common top-down coordinate system.

### Step 5: Render 3D Bounding Boxes

Annotated NuScenes objects are transformed and visualized relative to the ego vehicle.

---

## Visualization

The output visualization contains:

- Vehicle reference frame
- Camera locations
- Camera viewing directions
- Object positions
- 3D bounding boxes
- Top-down surround-view representation

Example visualization:

```text
           Front

    FL      F      FR

            ↑

Left ← Vehicle → Right

            ↓

    BL      B      BR

            Rear
```

---

## Key Concepts Demonstrated

### Camera Extrinsics

Defines where the camera is located relative to the vehicle.

### Camera Intrinsics

Defines how 3D points are projected into the image plane.

### Coordinate Transformations

Converts points between camera, ego, and global coordinate systems.

### Surround View Representation

Combines information from multiple cameras into a unified representation.

### 3D Bounding Box Visualization

Projects annotated objects into a common reference frame.

---

## Running the Demo

```bash
python surround_view.py
```

---

## Output

The script generates:

- 360° surround-view visualization
- Top-down camera layout
- Ego vehicle representation
- 3D bounding box visualization
- Multi-camera spatial alignment view

---

## Project Structure

```text
nuscenes-surround-view/
├── README.md
└── surround_view.py
```

---

## Future Improvements

Potential extensions include:

- BEV image stitching
- Multi-camera object fusion
- Occupancy grid generation
- Lane projection
- Object tracking
- Sensor fusion with LiDAR
- Real-time visualization
- Open3D interactive viewer

---

## References

- NuScenes Dataset
- NuScenes Devkit
- Bird's-Eye View (BEV) Perception
- Multi-Camera Geometry
- Autonomous Driving Perception Systems

---

## License

This project is intended for educational and research purposes.
