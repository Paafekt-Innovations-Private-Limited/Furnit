# PROVISIONAL PATENT APPLICATION NO. 6 OF 6

## TITLE OF INVENTION

### INTEGRATED MOBILE APPLICATION SYSTEM AND METHOD FOR ON-DEVICE PHOTOGRAPHIC ROOM SCANNING, METRIC SPATIAL RECONSTRUCTION, AND AUGMENTED REALITY FURNITURE VISUALIZATION

**INVENTOR:** Kishore Shivanna  
**APPLICANT:** Paafekt Inc.  
**FILING TYPE:** Provisional Patent Application  

---

## FIELD OF THE INVENTION

The present invention relates to mobile software application systems, and more particularly to an integrated, end-to-end mobile application system that combines guided photographic room scanning, on-device metric-accurate three-dimensional room reconstruction, interactive spatial measurement, and augmented reality furniture visualization into a unified user workflow, executing entirely on consumer mobile computing devices without requiring depth sensing hardware or cloud computing infrastructure.

---

## BACKGROUND OF THE INVENTION

The intersection of three-dimensional spatial computing and consumer retail—specifically the challenge of enabling consumers to make accurate furniture purchasing decisions informed by visualizations of how specific furniture will look and fit in their specific room—has been an active area of commercial development. Despite substantial investment by major technology companies and furniture retailers, the available solutions remain substantially incomplete:

### A. Fragmented Workflows

Existing tools address individual components of the furniture visualization workflow (room measurement OR 3D room scanning OR AR overlay) but do not integrate these capabilities into a seamless end-to-end user experience. Users must combine multiple apps, manually transfer measurements between them, and reconcile incompatible data formats. This fragmentation creates friction that substantially reduces practical utility.

### B. Hardware Exclusivity

The highest-quality end-to-end furniture visualization experiences (e.g., those offered by major furniture retailers' own applications) require LiDAR-equipped devices available only in premium device segments, excluding the majority of smartphone users globally from accessing the best experiences.

### C. Network Dependency

Applications not requiring LiDAR typically require network connectivity for cloud processing, preventing use in environments with poor connectivity and raising privacy concerns about transmission of home interior photography to commercial servers.

### D. Accuracy Insufficient for Purchasing

Existing applications that do work without LiDAR typically achieve insufficient measurement accuracy for confident purchasing decisions. A consumer purchasing a sofa needs confidence at the ±2 cm level; existing non-LiDAR apps provide accuracy of ±10–20 cm or worse.

### E. Absence of Integrated Retail Commerce

Existing room scanning and visualization tools are typically divorced from the product catalog and commerce infrastructure that would enable a user to move seamlessly from "does this sofa fit?" to "I want to buy this sofa."

The present invention provides the first commercially complete integration of all components—scanning, reconstruction, measurement, visualization, and commerce—into a single on-device application executable on standard consumer mobile hardware.

---

## SUMMARY OF THE INVENTION

The present invention is a mobile application for iOS and Android operating systems that enables a consumer user to: (1) photograph their room using an on-device guided scanning procedure; (2) obtain a metric-accurate three-dimensional model of the room from the photographs via entirely on-device computation; (3) interactively measure room dimensions with centimetre accuracy; (4) browse a furniture product catalog; (5) place selected furniture items into the room visualization and observe how they look at accurate scale in the specific room context; and (6) proceed to purchase furniture through integrated commerce functionality.

The application executes all spatial computing components (reconstruction, segmentation, measurement, rendering) entirely on the device, without transmitting room photographs or spatial data to external servers, supporting offline operation and protecting user privacy.

---

## DETAILED DESCRIPTION OF THE INVENTION

### 1. System Overview and End-to-End Architecture

FIG. 8 illustrates the end-to-end system architecture of the Paafekt mobile application. The system comprises the following major subsystems, operating in sequence through the user workflow:

- **(a) Guided Room Scanning Subsystem** — Guides the user through image capture.
- **(b) On-Device Reconstruction Subsystem** — Converts images to a metric 3D room model.
- **(c) Measurement Subsystem** — Enables spatial measurement queries against the room model.
- **(d) Furniture Catalog Subsystem** — Provides access to the furniture product database.
- **(e) Furniture Placement Subsystem** — Enables interactive placement of furniture in the room model.
- **(f) Augmented Reality Visualization Subsystem** — Renders composite views of placed furniture.
- **(g) Commerce Integration Subsystem** — Connects visualization to purchase workflow.
- **(h) Room Model Persistence Subsystem** — Saves and retrieves room models for later use.

### 2. User Workflow

FIG. 9 illustrates the complete user workflow through the Paafekt application.

#### Step 1 — Room Scanning

The user launches the application and selects "Scan a Room." The guided room scanning subsystem presents on-screen instructions and coverage indicators, guiding the user to capture a comprehensive set of room images over a scanning session typically lasting 30 to 90 seconds for a standard-sized room.

#### Step 2 — On-Device Reconstruction

Upon completion of the scanning session, the on-device reconstruction subsystem begins processing the captured images. Processing executes asynchronously in the background while the user interface displays a progress indicator and an incrementally-updating preview of the emerging 3D room model. On mid-range hardware, full reconstruction completes within approximately 30 to 120 seconds for a standard-sized room.

#### Step 3 — Room Model Review and Measurement

Upon reconstruction completion, the user is presented with the completed 3D room model in an interactive viewer. The user can inspect the model from any viewing angle, tap surfaces to see their dimensions, and access the measurement interface to query specific room dimensions.

#### Step 4 — Furniture Browsing and Selection

The user accesses the furniture catalog, browsing items by category, style, dimensions, price, or material. The catalog presents furniture items with product photography, specifications, and pricing.

#### Step 5 — Furniture Placement and Visualization

The user selects a furniture item for visualization. The application presents the room model in an interactive placement view. The user places the furniture item at a desired floor location. The application renders a photorealistic composite visualization of the furniture in the room at accurate scale.

#### Step 6 — Multiple Furniture Placement

The user can add multiple furniture items, building up a complete room arrangement visualization. The system maintains spatial consistency and collision detection between all placed items.

#### Step 7 — Purchase

From the furniture placement visualization, the user can view product details and proceed to purchase any visualized furniture item through the integrated commerce functionality.

#### Step 8 — Room Model Save and Share

The user can save the room model with its furniture arrangement for later retrieval and continuation. The user can capture still images or short video walkthroughs of the furnished room visualization for sharing.

### 3. Cross-Subsystem Data Model

The application maintains a unified internal data model shared across all subsystems. The central data model entities are:

#### A. RoomModel

Stores the metric-accurate 3D mesh, semantic surface labels, metric scale calibration parameters, camera pose set from reconstruction, room coordinate frame, computed dimensions, floor plan representation, and reconstruction quality metadata.

#### B. FurniturePlacement

Stores the furniture item identifier, metric position and orientation within the room coordinate frame, wall-snap status, and any placement constraints.

#### C. SceneState

Stores the current collection of FurniturePlacements within a RoomModel, representing the full room arrangement.

#### D. SessionState

Stores user session data including active RoomModel, active SceneState, selected furniture item, current view state, and interaction history.

The unified data model enables any subsystem to access any data entity it requires, and changes made by one subsystem are immediately visible to all others. This unified data architecture is what enables the seamless end-to-end workflow—the reconstruction subsystem writes to RoomModel, and the visualization subsystem reads from it without any manual data transfer step.

### 4. Privacy-Preserving Architecture

A key design principle of the Paafekt application is that room photography and spatial data remain entirely on the user's device. The application does not require user account creation for core functionality. The application does not transmit room photographs, spatial models, or measurement data to external servers. The entire spatial computing pipeline—scanning, reconstruction, segmentation, measurement computation, and rendering—executes on-device.

This privacy-preserving architecture is not merely a product feature but an architectural principle enforced at the software level: the spatial computing pipeline has no network communication interface for image or spatial data. Only voluntary user actions (such as sharing a visualization image or making a purchase) involve network communication.

### 5. Offline Operation Capability

Because all spatial computing executes on-device, the core functionality of the Paafekt application—scanning, reconstruction, measurement, furniture catalog browsing (for the locally-cached catalog), and visualization—operates fully offline without network connectivity. This enables use in environments with limited or absent internet access (including the interior of many buildings where mobile data coverage is poor).

### 6. Room Model Persistence and Multi-Session Use

The room model persistence subsystem stores completed room models in the device's persistent storage. Each stored room model includes: the full 3D mesh and semantic labels; the metric calibration parameters; the computed dimension set; and any furniture arrangements associated with the room.

Users can save multiple room models (corresponding to multiple rooms or multiple re-scans of the same room at different times). Returning to a saved room model, the user can continue furniture arrangement, modify existing placements, add new furniture items, and take new visualization captures—all without re-scanning the room.

### 7. Furniture Catalog Architecture

The furniture catalog subsystem provides access to furniture product data. In one embodiment, the catalog comprises a locally-stored subset of the full catalog (enabling offline browsing of commonly browsed items) and a network-accessed extension providing the full catalog when connectivity is available.

Each catalog entry stores: furniture item identifier; product name and description; product category; true metric dimensions (length, width, height); 3D geometric model for visualization; physically-based rendering material parameters; product photography images; pricing and availability data; and a retailer/manufacturer identifier linking to purchase flow.

The 3D geometric models and material parameters stored in the catalog are the inputs to the physically-based rendering pipeline, enabling accurate visualization. The true metric dimensions are used for scale-accurate placement within the room model.

### 8. Commerce Integration

The commerce integration subsystem connects the furniture visualization experience to purchase workflow. In one embodiment, from any furniture placement view, the user can access a product detail sheet for any placed furniture item, view additional product information and photography, and proceed to purchase through a secure in-application purchase flow or by opening the furniture retailer's website or application.

The integration tracks which furniture items the user has visualized in their room, enabling personalized recommendations and enabling furniture retailers to understand visualization-to-purchase conversion analytics.

### 9. Accessibility and Usability Design

The Paafekt application is designed for accessibility and ease of use by non-technical consumers:

#### A. Simplified Scanning Interface

The scanning interface presents intuitive visual cues rather than technical instructions. Coverage indicators use simple colored overlays rather than technical terminology.

#### B. Automatic Processing

After scanning, reconstruction is fully automatic. The user is not required to make any technical decisions during processing.

#### C. Measurement Display

Measurements are displayed in the user's preferred unit system (metric or imperial), automatically converted from the internal metric representation.

#### D. Undo/Redo

The furniture placement subsystem maintains a full undo/redo history, enabling non-destructive exploration of furniture arrangements.

#### E. Accessibility Standards Compliance

The application user interface implements platform accessibility standards enabling use by users with visual or motor impairments.

### 10. Cross-Platform Architecture

The Paafekt application is implemented for both iOS and Android mobile operating systems. The spatial computing pipeline—comprising neural network inference, visual odometry, depth fusion, segmentation, and rendering—is implemented in a platform-independent layer using cross-platform computational frameworks, enabling a single implementation of the core algorithms to execute on both platforms. Platform-specific layers handle native user interface components, camera access, NPU-specific neural inference APIs, and platform-specific GPU rendering APIs, encapsulated behind platform abstraction interfaces.

---

## CLAIMS

**Claim 1.** A mobile application system for furniture visualization, comprising: a guided room scanning module configured to guide a user through photographic capture of an interior room using a monocular camera of a mobile computing device; an on-device reconstruction module configured to generate a metric-accurate three-dimensional model of the interior room from the captured photographs, executing entirely on the mobile computing device; a measurement module configured to compute spatial measurements of the interior room from the three-dimensional model; a furniture catalog module configured to provide access to a database of furniture items with associated metric dimensions and three-dimensional models; a furniture placement module configured to place virtual representations of selected furniture items within the three-dimensional room model at user-specified locations with scale determined by stored metric dimensions; and a rendering module configured to generate composite visualizations of placed furniture within the room; wherein all spatial computing operations execute entirely on the mobile computing device without cloud offloading.

**Claim 2.** The system of claim 1, wherein the mobile computing device does not require a LiDAR sensor, structured-light sensor, time-of-flight sensor, or stereoscopic camera pair.

**Claim 3.** The system of claim 1, further comprising a room model persistence module configured to store the three-dimensional room model in persistent storage of the mobile computing device and retrieve it for subsequent use without re-scanning.

**Claim 4.** The system of claim 1, wherein the furniture catalog module maintains a locally-stored furniture catalog subset enabling offline browsing without network connectivity.

**Claim 5.** The system of claim 1, further comprising a commerce integration module configured to enable purchase of a visualized furniture item from within the application.

**Claim 6.** The system of claim 1, further comprising a unified internal data model shared across all modules, wherein data written by one module is accessible to all other modules without manual transfer.

**Claim 7.** The system of claim 1, wherein spatial computing operations do not transmit room photographic data or spatial model data to external servers, preserving user privacy.

**Claim 8.** The system of claim 1, wherein all core functionality including room scanning, reconstruction, measurement, furniture catalog browsing, and furniture visualization operates without network connectivity.

**Claim 9.** The system of claim 1, further comprising a scene state data model storing a collection of furniture placements within a room model representing a complete furniture arrangement.

**Claim 10.** The system of claim 1, further comprising a multi-session capability enabling a user to store multiple room models corresponding to multiple rooms or multiple scan sessions.

**Claim 11.** The system of claim 1, implemented for both iOS and Android mobile operating systems with a shared platform-independent spatial computing layer.

**Claim 12.** The system of claim 1, wherein the furniture placement module maintains an undo/redo history of furniture placement actions enabling non-destructive arrangement exploration.

**Claim 13.** The system of claim 1, wherein the rendering module generates a still image or video capture of the composite visualization for sharing.

**Claim 14.** The system of claim 1, wherein the guided room scanning module provides real-time visual coverage indicators overlaid on a live viewfinder display.

**Claim 15.** A computer-implemented method for integrated furniture visualization on a mobile computing device, comprising: guiding a user to capture a plurality of images of an interior room using a monocular camera; reconstructing a metric-accurate three-dimensional model of the interior room from the captured images entirely on the mobile computing device; providing a furniture catalog for browsing; receiving a user selection of a furniture item from the catalog; placing a virtual three-dimensional model of the furniture item within the reconstructed room model at a user-specified position with scale determined by stored metric dimensions; and rendering a composite visualization of the furniture item in the room; wherein the method executes entirely on the mobile computing device without cloud computation or depth sensing hardware.

**Claim 16.** The method of claim 15, further comprising computing spatial measurements of the interior room from the three-dimensional model with centimetre-level accuracy.

**Claim 17.** The method of claim 15, further comprising enabling purchase of the furniture item from within the application workflow.

**Claim 18.** The method of claim 15, further comprising storing the room model and furniture placement for retrieval in a subsequent session.

**Claim 19.** The method of claim 15, further comprising executing the method on mobile computing devices spanning a range of hardware capability without requiring premium or specialized hardware.

**Claim 20.** A non-transitory computer-readable medium storing instructions that, when executed by a processor of a mobile computing device, perform the method of claim 15.

---

## ABSTRACT

An integrated mobile application system and method for on-device furniture visualization. A guided scanning module directs monocular camera image capture of an interior room. An on-device reconstruction module generates a metric-accurate three-dimensional room model from captured images, executing entirely on device without cloud computation or depth sensing hardware. A measurement module computes centimetre-accurate spatial dimensions from the model. A furniture catalog module provides offline-capable access to furniture items with stored metric dimensions and three-dimensional geometric models. A furniture placement module positions virtual furniture within the room model at accurate physical scale. A rendering module produces photorealistic composite visualizations. Commerce integration enables in-app purchase of visualized furniture. Room model persistence enables multi-session use without re-scanning. All spatial computing executes on-device protecting user privacy and enabling offline operation on standard consumer mobile hardware without LiDAR or premium device requirements.
