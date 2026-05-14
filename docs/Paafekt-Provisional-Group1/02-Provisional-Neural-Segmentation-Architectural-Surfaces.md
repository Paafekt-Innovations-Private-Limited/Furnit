# PROVISIONAL PATENT APPLICATION NO. 2 OF 6

## TITLE OF INVENTION

### NEURAL OBJECT SEGMENTATION SYSTEM FOR ARCHITECTURAL SURFACE DECOMPOSITION FROM INDOOR PHOTOGRAPHY ON MOBILE DEVICES

**INVENTOR:** Kishore Shivanna  
**APPLICANT:** Paafekt Inc.  
**FILING TYPE:** Provisional Patent Application  

---

## FIELD OF THE INVENTION

The present invention relates to image segmentation systems and methods, and more particularly to on-device neural network-based semantic segmentation of indoor photographic images for the purpose of decomposing a room scene into its constituent architectural surfaces and objects, including floor, walls, ceiling, and furniture elements, for use in spatial measurement and augmented reality visualization applications.

---

## BACKGROUND OF THE INVENTION

Accurate identification and delineation of architectural surfaces within photographic images of interior spaces is a prerequisite for a wide range of valuable applications including: furniture visualization, interior design simulation, real estate documentation, accessibility assessment, and augmented reality spatial anchoring. The challenge of semantic segmentation of indoor scenes presents unique difficulties not encountered in general-purpose image segmentation:

### A. Surface Appearance Ambiguity

Interior architectural surfaces such as floors, walls, and ceilings frequently exhibit similar textures, colors, or patterns. White or neutral-colored walls and ceilings may be visually indistinguishable by color alone. Highly patterned floor tiles or area rugs may resemble textured wall coverings. Prior art general-purpose segmentation systems trained on diverse outdoor or mixed-domain datasets perform poorly on such ambiguous indoor surface appearances.

### B. Scale and Perspective Variation

Unlike outdoor scenes where the camera-to-subject distance is variable but objects are generally of predictable size, interior room photography presents extreme perspective foreshortening of floor and wall surfaces, and significant appearance variation of the same surface depending on viewing angle, distance, and field of view. A floor surface photographed from standing height appears very different from the same floor photographed from a low angle.

### C. Clutter and Occlusion

Real-world room environments contain furniture, personal effects, and other objects that partially occlude architectural surfaces. A wall partially obscured by a bookshelf, or a floor partially covered by furniture legs, must still be correctly identified and its extent estimated beyond the occlusion boundary.

### D. On-Device Computational Constraints

Mobile devices available to the mass market have computational capabilities—particularly regarding memory bandwidth, GPU throughput, and thermal limits—substantially below those of server-class hardware. Neural segmentation models of sufficient accuracy for interior surface decomposition are typically large in parameter count and computationally intensive, rendering them impractical for on-device inference on mid-range or older mobile hardware without significant architectural adaptation and optimization.

### E. Lack of 3D-Consistent Segmentation

Prior art 2D image segmentation systems operate on individual frames independently, without enforcing spatial consistency across multiple frames of the same scene from different viewpoints. When per-frame segmentation results are projected onto a shared 3D reconstruction, inconsistencies between frame segmentations produce artifacts and classification errors in the resulting 3D surface labeling.

---

## SUMMARY OF THE INVENTION

The present invention provides a system and method for on-device semantic segmentation of indoor photographic images into architectural surface classes, operating entirely on the mobile computing device without cloud connectivity, producing segmentation outputs at accuracy levels sufficient for furniture visualization, spatial measurement, and augmented reality applications.

In one embodiment, a multi-scale neural segmentation architecture, trained on a domain-specific dataset of interior room photography and optimized for on-device execution, processes individual frames to produce per-pixel class probability maps, which are subsequently fused across multiple frames using the 3D reconstruction coordinate frame to produce geometrically consistent 3D surface semantic labels.

---

## DETAILED DESCRIPTION OF THE INVENTION

### 1. Segmentation System Architecture

FIG. 4 illustrates the segmentation system architecture according to an embodiment of the present invention.

The segmentation system comprises: an image preprocessing module (200), a multi-scale feature extraction backbone (202), a semantic segmentation head (204), a multi-frame consistency fusion module (206), and a 3D surface label propagation module (208).

### 2. Neural Segmentation Architecture and On-Device Optimization

The multi-scale feature extraction backbone (202) is a hierarchical convolutional neural network that processes input frames at multiple resolutions, extracting features sensitive to both fine-grained surface texture information and coarse-grained scene layout information. The backbone architecture is designed for efficient execution on mobile neural processing hardware through: (a) use of depthwise-separable convolutional operations in place of standard convolutions, substantially reducing multiply-accumulate operation count while preserving representational capacity; (b) structured pruning of neural network weights to remove redundant or low-importance parameters without significant accuracy degradation; (c) post-training quantization of model weights and activations from 32-bit floating point to 8-bit integer representation, reducing memory footprint and increasing throughput on integer-capable neural processing hardware; (d) operator fusion combining adjacent neural network operations into single hardware-efficient kernels.

The semantic segmentation head (204) receives the multi-scale feature maps from the backbone and produces a per-pixel probability vector over the set of semantic classes. In one embodiment, the semantic class set comprises: floor, wall (generic), wall (primary, largest visible area), wall (secondary), ceiling, door, window, furniture (generic), and background/unknown.

### 3. Domain Adaptation for Indoor Photography

The neural segmentation model is trained on a domain-specific corpus of indoor room photography representative of the types of images encountered in consumer use of the application. The training corpus encompasses diverse characteristics including: (a) lighting conditions ranging from bright natural daylight to artificial incandescent and fluorescent lighting to mixed and low-light conditions; (b) room types including living rooms, bedrooms, dining rooms, kitchens, bathrooms, home offices, and commercial spaces; (c) surface materials and finishes including painted drywall, wallpaper, wood paneling, hardwood flooring, carpet, tile, concrete, and stone; (d) room contents including minimal unfurnished spaces and highly cluttered furnished spaces; (e) camera positions and orientations spanning the range expected during the guided scanning procedure.

Training with a domain-specific corpus, rather than a general-purpose indoor scene dataset, produces a model specialized for the specific visual distribution of images encountered in application use, substantially improving accuracy on the target domain while reducing model size requirements compared to general-purpose models trained to handle arbitrary visual domains.

### 4. Multi-Frame Segmentation Consistency

A critical innovation of the present invention is the multi-frame segmentation consistency mechanism implemented in the multi-frame consistency fusion module (206) and 3D surface label propagation module (208).

Per-frame segmentation predictions are not used independently. Instead, the system leverages the shared 3D reconstruction coordinate frame (obtained from the visual odometry and reconstruction pipeline) to aggregate segmentation predictions from all frames that observe each 3D surface point or voxel.

In one embodiment, for each 3D surface element (e.g., a face of the reconstructed mesh), the system identifies all camera frames in which that surface element is visible (i.e., not occluded and within the camera field of view). The per-pixel class probability vectors from each such frame, at the pixel location corresponding to the projection of the 3D surface element into that frame's image plane, are aggregated. The aggregation computes a weighted sum of class probability vectors, where weights account for: (a) the estimated surface coverage area at that pixel (larger surface patches at near-perpendicular incidence receive higher weight); (b) the estimated frame quality (sharpness, exposure quality); and (c) geometric incidence angle (surface patches viewed at more perpendicular angles receive higher weight than obliquely-viewed patches).

The aggregated probability vector for each 3D surface element is used to assign the final semantic class label through maximum a posteriori estimation. This multi-frame aggregation substantially increases robustness compared to single-frame classification: a surface patch that is partially obscured or poorly illuminated in one frame may receive a strong, reliable classification signal from another frame where the same surface is well-lit and unobscured.

### 5. Geometric Consistency Regularization

In addition to multi-frame aggregation, the semantic surface classifier applies a geometric consistency regularization that enforces agreement between the semantic label of a 3D surface patch and its geometric properties:

#### A. Normal-Based Regularization

Surface patches with computed 3D normal vectors oriented within a specified angular tolerance of the vertical direction (pointing up or down) are a priori more likely to be floor or ceiling than wall. Surface patches with normals oriented within a specified angular tolerance of horizontal are more likely to be wall. The geometric label prior is incorporated as an additive log-probability term in the semantic classification, biasing classification toward geometrically-consistent labels while allowing the appearance-based classification to override when the appearance signal is sufficiently strong.

#### B. Spatial Adjacency Regularization

Adjacent 3D surface patches are a priori more likely to share the same semantic class than to have different semantic classes. A Markov random field (MRF) smoothness term is applied to the 3D surface semantic labeling, penalizing semantic class transitions between adjacent surface patches weighted by their geometric similarity (co-planar, co-normal patches receive stronger smoothness pressure than patches with different normal orientations or curvature discontinuities between them).

#### C. Room Topology Constraints

The system enforces global room topology constraints: there must be exactly one floor region (the largest horizontal surface with a downward-facing normal), at most one ceiling region (the largest horizontal surface with an upward-facing normal), and one or more wall regions (vertical surfaces connecting the floor and ceiling regions). Candidate surface patches inconsistent with these global topology constraints are relabeled according to the global constraint.

### 6. Output and Downstream Interface

The semantic segmentation system produces as output: (a) a per-pixel semantic class label map for each input frame; (b) a 3D surface mesh with per-face semantic class labels; (c) aggregated per-class surface regions comprising the floor polygon, each wall polygon, and the ceiling polygon; (d) the geometric parameters of each semantic surface region including surface area, bounding dimensions, and plane equation parameters; and (e) the locations and dimensions of openings (doors, windows) detected within wall surfaces.

This output is consumed by the measurement, furniture placement, and visualization modules described in other patent applications of this bundle.

---

## CLAIMS

**Claim 1.** A computer-implemented method for semantic segmentation of interior room images, comprising: receiving a plurality of images of an interior room captured from a plurality of viewpoints using a monocular camera of a mobile computing device; executing, on a neural processing unit of the mobile computing device, a neural segmentation model to generate a per-pixel class probability map for each of the plurality of images; projecting the per-pixel class probability maps into a shared three-dimensional coordinate frame using estimated camera poses; aggregating class probability contributions from the plurality of images for each three-dimensional surface element; and assigning a semantic class label to each three-dimensional surface element based on the aggregated class probabilities; wherein all processing executes entirely on the mobile computing device without cloud connectivity.

**Claim 2.** The method of claim 1, wherein the semantic class labels comprise at least: floor, wall, and ceiling.

**Claim 3.** The method of claim 1, wherein the neural segmentation model employs depthwise-separable convolutional operations for computational efficiency on mobile neural processing hardware.

**Claim 4.** The method of claim 1, wherein the neural segmentation model is quantized to reduced-precision integer arithmetic for execution on mobile neural processing hardware.

**Claim 5.** The method of claim 1, wherein aggregating class probability contributions comprises weighting contributions from each image based on at least one of: surface coverage area, frame quality, and geometric incidence angle.

**Claim 6.** The method of claim 1, further comprising applying geometric consistency regularization that biases semantic class assignment based on surface normal orientation.

**Claim 7.** The method of claim 6, wherein the geometric consistency regularization biases surface patches with vertically-oriented normals toward floor or ceiling classification, and biases surface patches with horizontally-oriented normals toward wall classification.

**Claim 8.** The method of claim 1, further comprising applying spatial adjacency regularization that promotes consistent semantic class assignment between geometrically adjacent surface patches.

**Claim 9.** The method of claim 1, further comprising enforcing global room topology constraints requiring a single floor region, at most one ceiling region, and one or more wall regions.

**Claim 10.** The method of claim 1, further comprising detecting openings within wall surfaces and classifying detected openings as door openings or window openings.

**Claim 11.** The method of claim 1, wherein the neural segmentation model is trained on a domain-specific dataset of interior room photography.

**Claim 12.** The method of claim 1, further comprising generating a geometric parameter set for each semantic surface region comprising at least one of: surface area, bounding dimensions, and plane equation parameters.

**Claim 13.** The method of claim 1, wherein the estimated camera poses are obtained from a visual odometry procedure performed on the plurality of images.

**Claim 14.** A mobile computing system for semantic segmentation of interior room images, comprising: a monocular camera; a neural processing unit; one or more processors; and non-transitory computer-readable storage storing instructions to perform the method of claim 1.

**Claim 15.** A non-transitory computer-readable medium storing instructions that, when executed, perform the method of claim 1.

---

## ABSTRACT

A system and method for on-device semantic segmentation of interior room photographs, operating entirely on a mobile computing device without cloud connectivity. A neural segmentation model, optimized for mobile execution through depthwise-separable operations, quantization, and domain-specific training, generates per-pixel class probability maps for each captured frame. Multi-frame consistency fusion aggregates segmentation signals across all frames observing each three-dimensional surface element, using geometric incidence weighting. Geometric consistency regularization based on surface normals and spatial adjacency constraints, combined with global room topology enforcement, produces a semantically-labeled three-dimensional room surface model identifying floor, wall, ceiling, and architectural opening regions with their geometric parameters.
