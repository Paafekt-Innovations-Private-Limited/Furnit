# PROVISIONAL PATENT APPLICATION NO. 1 OF 6

## TITLE OF INVENTION

### ON-DEVICE THREE-DIMENSIONAL ROOM RECONSTRUCTION FROM MONOCULAR IMAGE SEQUENCES WITHOUT DEPTH SENSOR HARDWARE

**INVENTOR:** Kishore Shivanna  
**APPLICANT:** Paafekt Inc.  
**CORRESPONDENCE ADDRESS:** [Address of Applicant]  
**FILING TYPE:** Provisional Patent Application  

---

## FIELD OF THE INVENTION

The present invention relates to three-dimensional (3D) spatial reconstruction systems and methods, and more particularly to on-device, real-time 3D room reconstruction using monocular image capture sequences on mobile computing devices, without requiring dedicated depth-sensing hardware such as LiDAR sensors, structured-light projectors, or time-of-flight sensors.

---

## BACKGROUND OF THE INVENTION

### Prior Art and Its Limitations

The visualization of interior spaces in three dimensions has significant commercial value across industries including real estate, interior design, furniture retail, home renovation, and augmented reality (AR) entertainment. Existing approaches to 3D room reconstruction for consumer mobile applications suffer from one or more of the following critical deficiencies:

#### A. Hardware Dependency on Depth Sensors

The predominant class of commercially available mobile 3D reconstruction and AR measurement applications requires specialized depth-sensing hardware integrated into the mobile device. Such hardware includes, but is not limited to: LiDAR (Light Detection and Ranging) scanners, structured-light infrared projectors, time-of-flight (ToF) sensors, and stereo camera arrays. These depth sensors are present only in premium mobile device variants—for example, certain high-end smartphone product lines—representing a small fraction of the global installed base of mobile devices. Consequently, the overwhelming majority of consumers owning standard, mid-range, or older smartphones are excluded from using such applications. This hardware dependency creates an artificial and unnecessary barrier to access.

#### B. Cloud-Dependent Processing Architectures

Existing 3D reconstruction systems that do not rely on specialized hardware sensors typically require continuous or batch transmission of image data to remote cloud servers for computation. This architecture introduces multiple categories of failure: (i) the reconstruction cannot function without active network connectivity; (ii) latency inherent in network round-trips degrades the real-time or near-real-time user experience essential for practical room scanning; (iii) transmission of photographic data of users' private interior spaces raises significant privacy concerns; (iv) cloud processing costs create per-use economics that are unsustainable for low-cost or free-to-consumer application models; and (v) geographic regions with limited internet connectivity are excluded from use.

#### C. Inaccuracy of Monocular Depth Estimation Without Architectural Constraints

Prior attempts to estimate depth from a single monocular camera without specialized hardware have produced results insufficient in accuracy for practical measurement applications. Generic monocular depth estimation models, when applied to interior room environments without domain-specific architectural constraints, produce metric-scale ambiguities—meaning the reconstructed geometry is proportionally correct but lacks an absolute scale reference. This results in measurements that may be proportionally meaningful but are metrically inaccurate, rendering them unsuitable for furniture placement, space planning, or accurate interior measurement workflows.

#### D. Computational Infeasibility on Constrained Hardware

3D reconstruction algorithms of sufficient accuracy have historically required computational resources—GPU capabilities, memory bandwidth, processing cycles—available only in desktop or server computing environments. Attempts to port such algorithms to mobile devices have resulted either in unacceptable processing times (tens of minutes for a room scan) or in such aggressive architectural simplifications that measurement accuracy is sacrificed below practical utility thresholds.

#### E. Absence of Structural Semantic Understanding

Prior art 3D reconstruction systems treat the captured space as a generic geometric point cloud or mesh, without semantic understanding of the architectural components being reconstructed. A system that cannot distinguish wall surfaces from floor surfaces from furniture surfaces cannot generate the structured, semantically-organized room model needed for furniture visualization applications. Geometric reconstruction alone is insufficient; what is required is a system that simultaneously reconstructs geometry AND classifies that geometry into its structural components.

There exists, therefore, a long-felt and unmet need in the art for a 3D room reconstruction system that operates entirely on-device without cloud connectivity, functions on standard consumer mobile hardware without requiring specialized depth sensors, achieves centimetre-accurate metric reconstruction, and produces semantically-organized room models suitable for downstream furniture visualization and space planning workflows.

---

## SUMMARY OF THE INVENTION

The present invention provides a system and method for performing complete three-dimensional room reconstruction using only a standard monocular camera as the image capture hardware, executing all computational processes entirely on the mobile device without cloud connectivity, and producing metric-accurate 3D room models with centimetre-level precision.

In one embodiment, the invention comprises a mobile software application executing on a consumer mobile computing device equipped with a standard rear-facing monocular camera, wherein the application guides a user through a room scanning procedure, captures a sequence of overlapping still images or video frames from multiple vantage points within the room, and processes said images entirely on the device to generate a complete three-dimensional model of the room including dimensions, floor plan, wall surfaces, ceiling height, and spatial relationships between all detected surfaces.

In another embodiment, the invention employs a multi-stage processing pipeline that sequentially performs: (i) feature extraction and matching across the image sequence; (ii) camera pose estimation for each captured frame; (iii) dense depth estimation per frame using a domain-adapted neural inference model operating on the device's neural processing hardware; (iv) multi-view depth fusion to generate a consistent 3D point representation; (v) scale disambiguation using detected architectural constraints; and (vi) mesh generation and semantic surface labeling.

A key aspect of the invention is the scale disambiguation mechanism, which leverages the co-planarity constraints of room floor and ceiling surfaces, along with detectable architectural features such as door frames and standard-dimension structural elements, to resolve the metric-scale ambiguity inherent in monocular reconstruction, enabling centimetre-accurate absolute measurements without any depth sensing hardware.

---

## DETAILED DESCRIPTION OF THE INVENTION

### 1. System Architecture Overview

FIG. 1 illustrates a high-level block diagram of the on-device 3D room reconstruction system according to an embodiment of the present invention. The system executes entirely within the application runtime environment on a mobile computing device (100), comprising a central processing unit (CPU) (102), a graphics processing unit (GPU) (104), a dedicated neural processing unit (NPU) or equivalent hardware accelerator (106), system memory (108), persistent storage (110), a monocular image sensor (112), and a display (114). No external network communication is required during reconstruction.

The software system comprises: an image acquisition module (120), a feature extraction module (122), a visual odometry module (124), a neural depth estimation module (126), a depth fusion module (128), a scale resolver module (130), a mesh generation module (132), and a semantic surface classifier (134).

### 2. Image Acquisition and Guided Scanning

FIG. 2 illustrates the image acquisition and guided scanning procedure according to an embodiment of the invention.

The image acquisition module (120) presents a real-time viewfinder display to the user and guides the user through a structured scanning procedure. In one embodiment, the scanning procedure instructs the user to: (a) begin at the center of the room; (b) execute a slow, continuous pan of the monocular camera across each wall surface from approximately eye-level height; (c) capture additional frames tilted downward toward the floor and upward toward the ceiling; and (d) capture frames from multiple positions within the room to ensure multi-directional coverage of all surface points.

The image acquisition module (120) monitors incoming camera frames in real-time, computing a frame selection score for each candidate frame based on: (i) feature richness (number of detectable visual interest points); (ii) motion magnitude relative to previously selected frames (ensuring adequate baseline for triangulation while avoiding motion blur); (iii) coverage overlap with the set of previously selected frames; and (iv) focus and exposure quality metrics. Frames falling below a minimum quality threshold are discarded. Frames that do not increase geometric coverage above a minimum marginal coverage threshold are also discarded. The resulting selected frame set represents a sparse, high-quality, geographically distributed sampling of the room's surface geometry.

In one embodiment, the acquisition module provides real-time visual feedback to the user overlaid on the viewfinder, indicating (a) areas of the room that have been sufficiently covered by selected frames (shown in one color or visual indicator), (b) areas requiring additional coverage (shown in another color or indicator), and (c) the current frame quality score. This guided acquisition procedure ensures that the resulting frame set is optimized for the downstream reconstruction pipeline.

### 3. Feature Extraction and Matching

The feature extraction module (122) processes each selected frame to identify and describe a set of local visual features. In one embodiment, a neural network-based keypoint detector and descriptor operates on each frame, producing a set of keypoint locations and associated high-dimensional descriptor vectors that are invariant to illumination changes, viewpoint changes, and scale variations. The neural network executing this function is quantized to reduced numerical precision and optimized for execution on the device's neural processing hardware (106), enabling per-frame processing within the time budget required for near-real-time operation on mid-range consumer devices.

The feature matching sub-module receives the descriptor sets from each pair of frames to be matched and computes pairwise feature correspondences using approximate nearest-neighbor matching in descriptor space. Geometrically inconsistent matches are rejected using a robust estimation procedure that fits a geometric transformation model to the correspondence set and identifies and removes outlier correspondences.

### 4. Visual Odometry and Camera Pose Estimation

The visual odometry module (124) receives the set of validated feature correspondences across frame pairs and constructs a camera pose graph. Each node in the pose graph represents a captured frame, and each edge represents a set of validated correspondences between a pair of frames with an associated relative pose estimate.

In one embodiment, the relative pose between a pair of frames is estimated from the validated correspondences using the five-point algorithm or equivalent minimal solver, producing the rotation and translation (up to scale) relating the two camera positions. The collection of relative poses is optimized globally through a bundle adjustment procedure that simultaneously refines all camera poses and the 3D positions of all triangulated feature points to minimize the total reprojection error across all frames and correspondences. The bundle adjustment is executed as a sparse nonlinear least-squares optimization, with the sparsity pattern of the problem exploited to render it computationally tractable on the device CPU (102) within acceptable time bounds.

The output of the visual odometry module is a set of camera poses—each comprising a rotation matrix and translation vector—in a consistent coordinate frame, along with a sparse 3D point cloud of triangulated feature points. At this stage, the reconstruction is metrically consistent but lacks absolute scale.

### 5. Neural Depth Estimation

The neural depth estimation module (126) processes each selected frame through an on-device neural inference model to produce a dense per-pixel depth estimate. In one embodiment, the neural inference model is a convolutional neural network architecture trained on interior room datasets and domain-adapted to handle diverse consumer-grade indoor photography conditions including: varying illumination (natural light, artificial light, mixed lighting); diverse room sizes from small closets to large open-plan living spaces; diverse surface materials including specular surfaces, textureless surfaces, and patterned surfaces; and clutter from furniture and personal effects.

The neural inference model is compiled and quantized for execution on the device's neural processing hardware (106), achieving per-frame inference times compatible with the requirements of the on-device reconstruction pipeline. The output is an affine-invariant depth map for each frame—a 2D array of values proportional to scene depth at each pixel, where the proportionality constant and additive offset are initially unknown (the metric-scale ambiguity).

### 6. Multi-View Depth Fusion

The depth fusion module (128) receives the per-frame dense depth maps from the neural depth estimation module and the calibrated camera poses from the visual odometry module. It fuses these inputs into a consistent volumetric representation of the room geometry.

In one embodiment, the depth fusion procedure operates by: (a) back-projecting each per-frame depth map into 3D space using the estimated camera pose and intrinsic parameters of the monocular camera; (b) identifying voxels in a volumetric grid that are observed by multiple camera frames; (c) computing a fused depth value for each observed voxel as a weighted average of the contributing frame depth estimates, where weights account for estimated depth reliability, incidence angle, and visibility; and (d) resolving conflicts between depth estimates from different frames using robust statistical estimation.

The fused volumetric representation is then converted to a surface representation—in one embodiment, a triangulated mesh—using a surface extraction algorithm operating on the implicit surface representation stored in the volumetric grid.

### 7. Scale Disambiguation — Key Inventive Element

The scale resolver module (130) implements the critical scale disambiguation mechanism that elevates the reconstruction from metrically-proportional to metrically-accurate (centimetre-level precision). This module is a key inventive contribution of the present invention.

FIG. 3 illustrates the scale disambiguation pipeline.

The scale disambiguation approach exploits multiple sources of metric constraints available within a standard interior room environment, without requiring any depth sensing hardware:

#### A. Floor-Ceiling Parallelism Constraint

In any habitable room, the floor and ceiling surfaces are parallel horizontal planes. The reconstructed mesh contains surface patches that have been semantically classified as floor and ceiling (see Section 8 below). The vertical separation between the floor plane and ceiling plane, expressed in the reconstruction's internal coordinate units, is measured. A statistical prior over residential and commercial ceiling heights (with a standard distribution centered on approximately 2.4 to 3.0 meters) is applied to generate a probability distribution over possible scale factors. The maximum a posteriori scale factor is selected as the initial scale estimate.

#### B. Architectural Feature Constraints

The image frames captured during scanning are analyzed for the presence of architectural features with known standard dimensions. In one embodiment, detected door frames are used: a door frame detector applies a neural object detection model to identify door frame locations in frames, and the height and width of detected door frames are measured in the reconstruction's internal coordinate units. The ratio of standard door dimensions (approximately 2.0 meters height, 0.8 meters width) to the measured internal-unit dimensions provides an additional scale estimate.

#### C. Inertial Sensor Integration

In one embodiment, the scale disambiguation is supplemented by integration of signals from the mobile device's inertial measurement unit (IMU), comprising accelerometer and gyroscope sensors. Known gravitational acceleration magnitudes from the accelerometer, combined with tracked motion dynamics during the scanning procedure, provide additional metric constraints that are fused with the geometric scale estimates.

#### D. Multi-Constraint Fusion

The scale estimates derived from each constraint source are fused using a weighted Bayesian estimation procedure. The final metric scale factor is applied globally to the reconstruction, converting it from internal coordinate units to meters with centimetre-level precision.

### 8. Semantic Surface Classification

The semantic surface classifier (134) processes the reconstructed 3D mesh to assign a semantic class label to each surface patch. In one embodiment, the semantic classes include: floor, wall (per wall surface, individually labeled), ceiling, window opening, door opening, and generic object (for furniture or other non-structural elements captured in the scan).

Classification operates by combining: (a) geometric cues—surface normal orientation (horizontal normals for floor and ceiling, vertical normals for walls); surface area and connectivity; and (b) appearance cues—neural image segmentation predictions projected from the 2D frame captures onto the 3D surface using the calibrated camera poses.

The output is a semantically-labeled 3D mesh wherein each structural surface of the room is individually identified and labeled, enabling downstream queries such as "floor area," "wall area," "room dimensions," and spatial relationships between surfaces.

### 9. Output Room Model

The reconstruction pipeline produces a structured room model comprising: (a) a metric-accurate 3D mesh of all detected room surfaces; (b) a semantic labeling of each surface patch; (c) a floor plan representation (2D projection of the room geometry onto the horizontal plane); (d) measured dimensions including room length, width, height, and the positions and dimensions of openings (doors, windows); and (e) coordinate frames aligned to the room's structural axes (the principal horizontal and vertical directions of the room).

This structured room model is stored in persistent storage (110) on the device and made available to the furniture placement and visualization modules described in the other patent applications in this bundle.

---

## CLAIMS

**Claim 1.** A computer-implemented method for three-dimensional room reconstruction, comprising: capturing, using a monocular image sensor of a mobile computing device, a plurality of images of an interior room from a plurality of positions and orientations; extracting visual features from each of the plurality of images; estimating camera poses for each of the plurality of images based on feature correspondences between images; estimating, using an on-device neural inference model, a depth map for each of the plurality of images; fusing the plurality of depth maps with the estimated camera poses to generate a three-dimensional surface representation of the interior room; resolving a metric scale of the three-dimensional surface representation using at least one architectural constraint detected within the plurality of images; and outputting a metric-accurate three-dimensional model of the interior room; wherein all steps of the method execute entirely on the mobile computing device without transmission of image data to an external server.

**Claim 2.** The method of claim 1, wherein the monocular image sensor is the sole depth-sensing input to the method, and wherein no LiDAR sensor, structured-light sensor, time-of-flight sensor, or stereoscopic camera pair is used.

**Claim 3.** The method of claim 1, wherein resolving the metric scale comprises detecting a floor surface and a ceiling surface in the three-dimensional surface representation, measuring a separation distance between the floor surface and the ceiling surface in internal reconstruction units, and applying a statistical prior distribution over room height dimensions to determine a metric scale factor.

**Claim 4.** The method of claim 1, wherein resolving the metric scale further comprises detecting one or more architectural features of standard known dimensions within the plurality of images, measuring dimensions of the detected architectural features in internal reconstruction units, and computing a metric scale factor from the ratio of standard known dimensions to measured internal-unit dimensions.

**Claim 5.** The method of claim 4, wherein the architectural features of standard known dimensions comprise door frames.

**Claim 6.** The method of claim 1, wherein resolving the metric scale further comprises integrating signals from an inertial measurement unit of the mobile computing device with geometric scale estimates to produce a fused metric scale estimate.

**Claim 7.** The method of claim 1, wherein estimating camera poses comprises constructing a camera pose graph and performing bundle adjustment to jointly optimize camera poses and three-dimensional positions of triangulated feature points.

**Claim 8.** The method of claim 1, wherein extracting visual features comprises executing a quantized neural network keypoint detector and descriptor on the mobile computing device's neural processing hardware.

**Claim 9.** The method of claim 1, wherein estimating a depth map comprises executing a convolutional neural network trained on interior room image datasets and adapted for consumer mobile device photography conditions.

**Claim 10.** The method of claim 9, wherein the convolutional neural network is quantized to reduced numerical precision for efficient execution on a neural processing unit of the mobile computing device.

**Claim 11.** The method of claim 1, further comprising semantically classifying surfaces of the three-dimensional surface representation into a plurality of structural categories comprising at least: floor, wall, and ceiling.

**Claim 12.** The method of claim 11, wherein semantically classifying surfaces comprises applying geometric cues based on surface normal orientations and applying appearance cues based on projected neural image segmentation predictions.

**Claim 13.** The method of claim 1, further comprising guiding a user through an image capture procedure by providing real-time visual feedback indicating spatial coverage of the interior room by the plurality of captured images.

**Claim 14.** The method of claim 13, wherein the real-time visual feedback comprises visual indicators overlaid on a live viewfinder display distinguishing captured regions from uncaptured regions of the interior room.

**Claim 15.** The method of claim 1, further comprising selecting a subset of frames from a captured image sequence based on frame quality metrics comprising at least one of: feature richness, motion magnitude, coverage overlap, focus quality, and exposure quality.

**Claim 16.** The method of claim 1, wherein the metric-accurate three-dimensional model achieves centimetre-level dimensional accuracy for room dimensions.

**Claim 17.** The method of claim 1, further comprising generating a two-dimensional floor plan representation from the metric-accurate three-dimensional model.

**Claim 18.** A mobile computing system for three-dimensional room reconstruction, comprising: a monocular image sensor; one or more processors; a neural processing unit; and non-transitory computer-readable storage medium storing instructions that, when executed by the one or more processors, perform the method of claim 1.

**Claim 19.** The system of claim 18, wherein the mobile computing system is a smartphone or tablet computer.

**Claim 20.** A non-transitory computer-readable medium storing instructions that, when executed by a processor of a mobile computing device equipped with a monocular image sensor, perform the method of claim 1.

---

## ABSTRACT

A system and method for on-device three-dimensional room reconstruction using only a monocular camera of a standard consumer mobile computing device, without requiring LiDAR sensors, depth sensors, or cloud computing. The method captures a plurality of room images from multiple viewpoints, extracts visual features, estimates camera poses via visual odometry and bundle adjustment, estimates dense per-pixel depth maps using an on-device neural inference model, fuses depth maps into a unified three-dimensional surface representation, resolves metric scale using architectural constraints including floor-ceiling parallelism and standard-dimension architectural features, and semantically classifies surfaces into structural categories including floor, wall, and ceiling. All computation executes entirely on-device, producing centimetre-accurate three-dimensional room models on standard consumer hardware without network connectivity requirements.
