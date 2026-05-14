# PROVISIONAL PATENT APPLICATION NO. 4 OF 6

## TITLE OF INVENTION

### AUGMENTED REALITY FURNITURE PLACEMENT AND VISUALIZATION PIPELINE FOR METRIC-ACCURATE SPATIAL INTEGRATION ON MOBILE DEVICES

**INVENTOR:** Kishore Shivanna  
**APPLICANT:** Paafekt Inc.  
**FILING TYPE:** Provisional Patent Application  

---

## FIELD OF THE INVENTION

The present invention relates to augmented reality visualization systems and methods on mobile computing devices, and more particularly to systems and methods for placing, rendering, and interactively manipulating virtual three-dimensional representations of furniture items within a metric-accurate three-dimensional model of an interior room, producing photorealistic composite visualizations that accurately reflect the true spatial scale and appearance of the furniture as it would appear in the physical room.

---

## BACKGROUND OF THE INVENTION

The ability to visualize furniture in a specific room before purchase is a commercially significant capability that reduces returns, increases purchase confidence, and improves customer satisfaction. Existing approaches suffer from critical limitations:

### A. Inaccurate Scale

The most common category of mobile furniture visualization applications overlays 2D product images or low-fidelity 3D models onto a live camera feed without metric-accurate spatial calibration. The resulting visualizations display furniture at approximately correct scale but with sufficient inaccuracy (often ±15–25%) that users cannot rely on them for purchasing decisions requiring knowledge of whether furniture will physically fit in a space.

### B. Lack of Room Integration

Furniture visualizations that do not incorporate a full 3D model of the room cannot correctly handle spatial relationships between the virtual furniture and room geometry: the furniture does not correctly occlude or be occluded by walls, cannot be shown touching the floor at the correct position, and cannot reflect room lighting conditions.

### C. LiDAR Dependency for Spatial Anchoring

Applications that do achieve metric-accurate furniture visualization typically rely on LiDAR sensors for precise floor plane detection and object placement, limiting them to premium hardware.

### D. Unrealistic Rendering

Many furniture visualization applications use simplified shading models that produce obviously synthetic-looking renderings, reducing user confidence in the visualization's validity as a representation of how the furniture will actually appear.

---

## SUMMARY OF THE INVENTION

The present invention provides a system and method for augmented reality furniture placement and visualization that achieves metric-accurate spatial integration of virtual furniture items within a reconstructed three-dimensional room model on a standard consumer mobile device, without LiDAR or cloud dependency, and produces photorealistic composite visualizations showing the furniture at accurate scale in the correct room context.

---

## DETAILED DESCRIPTION OF THE INVENTION

### 1. Furniture Visualization Pipeline Architecture

FIG. 6 illustrates the furniture visualization pipeline. The pipeline comprises: a furniture model library (400), a furniture placement engine (402), a real-time rendering engine (404), an occlusion handling module (406), a room lighting estimation module (408), a camera pose tracker (410), and a visualization compositor (412).

### 2. Furniture Model Library

The furniture model library (400) stores three-dimensional geometric models of furniture items, each comprising: (a) a triangulated mesh representation of the furniture geometry; (b) physically-based rendering (PBR) material parameters for each surface of the furniture, including albedo (base color), roughness, metallic fraction, normal map, and ambient occlusion map; (c) metric dimensions of the furniture item (length, width, height in meters); and (d) furniture category classification and associated metadata.

In one embodiment, furniture models in the library are sourced from furniture manufacturers and retailers, processed into a standardized internal format, and stored in the device's persistent storage for offline access. The metric dimensions stored in the furniture model database are the physical dimensions of the actual furniture item, enabling the visualization system to render the virtual furniture at precisely its true physical scale within the metric-accurate room model.

### 3. Metric-Accurate Furniture Placement

The furniture placement engine (402) enables the user to place a selected furniture item within the reconstructed room model at a user-specified location. The placement workflow is as follows:

#### A. Floor Position Selection

The user selects a floor position for the furniture item by touching the display at the desired location. The system ray-casts from the touch point through the virtual camera into the 3D room model, identifies the intersection with the semantic floor surface, and computes the metric coordinates of the selected floor position in the room's coordinate frame.

#### B. Scale-Accurate Positioning

The selected furniture item's 3D model is positioned at the selected floor position with the furniture base plane coinciding with the reconstructed floor surface. Because the room model is metric-accurate (calibrated to centimetre precision) and the furniture model dimensions are the true physical dimensions of the furniture item, the furniture is automatically rendered at correct physical scale relative to the room—a piece of furniture 1.8 meters wide will occupy precisely 1.8 meters in the visualized room.

#### C. Interactive Position Adjustment

The user adjusts the furniture position by dragging on the display, which translates the furniture item along the floor plane of the room model. The furniture position is constrained to the detected floor surface, preventing the furniture from being placed floating above the floor or embedded below it.

#### D. Interactive Rotation

The user rotates the furniture item about a vertical axis by applying a rotation gesture (e.g., two-finger rotate). The furniture rotates about the vertical axis passing through its centroid while maintaining contact with the floor plane.

#### E. Wall Proximity Detection and Snapping

As the user drags the furniture item toward a wall surface, the system detects proximity between the furniture model's bounding geometry and the reconstructed wall surface. At a configurable proximity threshold, the furniture snaps to a position flush against the wall surface. This assists accurate against-wall placement and provides haptic feedback to the user upon wall contact.

#### F. Collision Detection

The placement engine detects spatial overlap (collision) between the placed furniture item and other placed furniture items or detected fixed architectural features (e.g., fireplace surrounds, room columns). When a collision is detected, the system provides visual feedback (e.g., highlighting the colliding regions) and optionally constrains placement to prevent overlap.

### 4. Photorealistic Rendering

The real-time rendering engine (404) renders the composite visualization of the virtual furniture within the room at a frame rate consistent with smooth interactive operation on the mobile device (targeting 30 or more frames per second).

#### A. Physically-Based Rendering

The rendering engine implements a physically-based rendering pipeline that models the interaction of room lighting with furniture surfaces using physically-accurate bidirectional reflectance distribution function (BRDF) models. The BRDF model simulates the realistic appearance of surface materials including wood grain, upholstered fabric, metal, glass, and painted surfaces, producing renderings that accurately represent the visual appearance of the furniture in the specific lighting conditions of the photographed room.

#### B. Room Lighting Estimation

The room lighting estimation module (408) analyzes the photographic images captured during room scanning to estimate the lighting environment of the room, including: (i) the positions, intensities, and color temperatures of detectable light sources; (ii) the ambient indirect illumination level (from light bouncing off room surfaces); and (iii) the directional distribution of incident illumination at the furniture placement location. In one embodiment, an environment map (spherical representation of the incident illumination from all directions) is estimated from the room photographs and used to illuminate the virtual furniture models.

#### C. Shadow Generation

The rendering engine generates shadow projections from the virtual furniture items onto the room's floor and wall surfaces. Shadows are computed using the estimated room lighting environment, producing shadow shapes and intensities consistent with the estimated light source positions and furniture geometry.

#### D. Material Rendering Optimization

The physically-based rendering computations are implemented to execute within the thermal and computational budget of mid-range mobile hardware at the required frame rate. In one embodiment, rendering employs screen-space approximations to computationally expensive effects (ambient occlusion, indirect illumination) that approximate the full computation with lower computational cost and acceptable visual quality degradation.

### 5. Occlusion Handling

The occlusion handling module (406) manages the spatial occlusion relationships between virtual furniture and the real room environment. In one embodiment, the system renders virtual furniture items occluded by reconstructed room geometry (e.g., a furniture item placed in a corner is partially occluded by the walls when viewed from certain angles). The occlusion is computed using the reconstructed 3D room mesh as an occluder in the rendering pipeline.

Additionally, the system handles occlusion of virtual furniture by real-world objects present in the room (e.g., a real bookshelf between the camera and the placed virtual sofa). In one embodiment, real-world foreground objects detected in the live camera feed are segmented and composited over the virtual furniture rendering, producing the correct occlusion appearance.

### 6. Real-Time Camera Pose Tracking

The camera pose tracker (410) maintains a continuous estimate of the mobile device's camera position and orientation relative to the reconstructed room model in real-time during the interactive furniture visualization phase. This tracking enables the visualization to update correctly as the user moves the device to view the furniture from different angles.

In one embodiment, real-time tracking is implemented using a combination of: (a) inertial measurement unit (IMU) data from the device's accelerometer and gyroscope for high-frequency orientation tracking; and (b) visual localization—matching features detected in the current live camera frame against the reconstructed room model and stored keyframe images to update the metric position estimate.

---

## CLAIMS

**Claim 1.** A computer-implemented method for augmented reality furniture visualization, comprising: receiving a metric-accurate three-dimensional model of an interior room generated from monocular camera images; receiving a selection of a furniture item having stored metric dimensions; receiving a user-specified placement location within the interior room; placing a three-dimensional model of the furniture item at the placement location with scale determined by the stored metric dimensions and the metric calibration of the room model; rendering a composite visualization of the furniture item within the room at the placement location; and displaying the composite visualization on a display of a mobile computing device; wherein all processing executes entirely on the mobile computing device.

**Claim 2.** The method of claim 1, wherein placing the three-dimensional model comprises constraining the furniture model to a reconstructed floor surface of the room model.

**Claim 3.** The method of claim 1, wherein placing the three-dimensional model comprises enabling user manipulation including at least one of: translation along the room floor, rotation about a vertical axis, and wall-proximity snapping.

**Claim 4.** The method of claim 1, further comprising detecting proximity between the furniture model and a reconstructed wall surface and snapping the furniture model to a flush-against-wall position upon detecting proximity below a threshold.

**Claim 5.** The method of claim 1, further comprising detecting spatial collision between the furniture model and at least one of: other placed furniture items and fixed architectural features.

**Claim 6.** The method of claim 1, wherein rendering comprises implementing a physically-based rendering pipeline using estimated room lighting conditions.

**Claim 7.** The method of claim 6, wherein estimated room lighting conditions are derived from photographic images captured during a room scanning procedure.

**Claim 8.** The method of claim 6, wherein rendering further comprises generating shadow projections from the furniture model onto room surfaces based on estimated light source positions.

**Claim 9.** The method of claim 1, further comprising handling occlusion of the furniture model by reconstructed room geometry.

**Claim 10.** The method of claim 1, further comprising tracking a real-time camera pose of the mobile device relative to the room model and updating the composite visualization responsive to tracked camera pose changes.

**Claim 11.** The method of claim 10, wherein tracking the real-time camera pose combines inertial measurement unit data with visual localization against the reconstructed room model.

**Claim 12.** The method of claim 1, wherein the composite visualization accurately represents the physical scale of the furniture item within the room to within two centimetres.

**Claim 13.** The method of claim 1, further comprising enabling simultaneous placement of a plurality of furniture items within the room model and rendering a composite visualization showing all placed furniture items in their respective positions.

**Claim 14.** The method of claim 1, wherein the furniture item is selected from a product catalog stored on the mobile computing device for offline access.

**Claim 15.** The method of claim 1, further comprising generating a still image or video capture of the composite visualization for sharing.

**Claim 16.** A mobile computing system for augmented reality furniture visualization, comprising: a monocular camera; a display; one or more processors; and non-transitory computer-readable storage storing a furniture model library and instructions to perform the method of claim 1.

**Claim 17.** A non-transitory computer-readable medium storing instructions that, when executed, perform the method of claim 1.

---

## ABSTRACT

A system and method for augmented reality furniture placement and visualization on a standard consumer mobile computing device without depth sensors or cloud processing. Virtual furniture items with stored true metric dimensions are placed on a metric-accurate reconstructed floor surface of a three-dimensional room model generated from monocular camera images. A physically-based rendering pipeline illuminated by lighting conditions estimated from room photographs generates photorealistic composite visualizations of furniture within the room context, with accurate scale, shadow generation, and occlusion handling. Interactive controls enable floor-plane translation, vertical-axis rotation, and wall-snapping. Real-time camera pose tracking updates the visualization as the user's viewpoint changes.
