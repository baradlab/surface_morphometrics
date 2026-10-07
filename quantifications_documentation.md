# Morphometrics Quantifications Documentation
---

### `Index`
- **Type:** Integer  
- **Description:**  
  An identifier for each triangle in the mesh. Serves as the primary key for a given surface and is used to cross-reference data (e.g., in the `[label]_neighbor_index` column). The index is set when pycurv is run; any filtering afterward (such as patch extraction or edge filtering) does not reindex.

---

### `Area`
- **Type:** Float  
- **Description:**  
  The surface area of the individual triangle. Important for quantitative analysis, as it allows for area-weighted statistical calculations (e.g., histograms). By weighting metrics by triangle area, larger membrane regions contribute proportionally more to overall statistics, providing a more accurate representation of the surface's properties.

---

### `min_curvature`
- **Type:** Float  
- **Description:**  
  The minimum absolute curvature value at each point on the surface.

---

### `max_curvature`
- **Type:** Float  
- **Description:**  
  The maximum absolute curvature value at each point on the surface.

---

### `Gauss_curvature`
- **Type:** Float  
- **Description:**  
  The Gaussian curvature measured at each point on the surface, prior to vector voting. Calculated as the product of the two principal curvatures at that point. **Note:** Use `gauss_curvature_vv` instead for analysis, as it provides more robust measurements.

---

### `mean_curvature`
- **Type:** Float  
- **Description:**  
  The mean curvature value computed at each point on the surface, before any vector voting or smoothing operations. Mean curvature is calculated as the average of the curvatures in all directions at a point. **Note:** Use `mean_curvature_vv` instead for analysis, as it provides more robust measurements.

---

### `kappa_1`
- **Type:** Float  
- **Description:**  
  The first principal curvature value at each point on the surface.

---

### `kappa_2`
- **Type:** Float  
- **Description:**  
  The second principal curvature value at each point on the surface.

---

### `gauss_curvature_vv`
- **Type:** Float  
- **Description:**  
  **PREFERRED VERSION** - The Gaussian curvature, calculated as the product of the two principal curvatures (`kappa_1 * kappa_2`) after vector voting. The sign indicates local surface shape:
  - **Positive (> 0):** Surface is elliptical or "bowl-shaped"
  - **Negative (< 0):** Surface is hyperbolic or "saddle-shaped"
  - **Zero (= 0):** Surface is flat in at least one direction

---

### `mean_curvature_vv`
- **Type:** Float  
- **Description:**  
  **PREFERRED VERSION** - The mean curvature, calculated as the average of the two principal curvatures: (`kappa_1 + kappa_2`) / 2 after vector voting. Describes the average bending of the surface at a specific point:
  - **Positive (> 0):** The surface is, on average, curving inwards (concave)
  - **Negative (< 0):** The surface is, on average, curving outwards (convex)
  - **Zero (= 0):** The surface is a minimal surface

---

### `Orientation_class`
- **Description:**  
  A classification for each triangle describing the consistency of normal vectors within its local geodesic neighborhood. Controlled by the epsilon and eta parameters in pycurv functions, which set thresholds for the classes.
- **Classes:**
  - **Class 1 (Preferred Orientation):** Neighboring triangle normals are highly consistent and point in a similar direction, indicating a smooth, well-defined surface patch. Most triangles on a quality mesh belong to this class.
  - **Class 2 (Crease Junction):** Neighboring normals are aligned along a line or "crease," indicating a sharp edge or fold.
  - **Class 3 (No Preferred Orientation):** Neighboring normals are inconsistent and point in random directions, indicating high, complex curvature, noise, or mesh artifact.

---

### `shape_index_VV`
- **Description:**  
  **PREFERRED VERSION** - A continuous value ranging from -1 to +1 that describes the local shape of the surface at each triangle, derived from principal curvatures (`kappa_1` and `kappa_2`) after vector voting.
- **Values:**
  - **-0.75:** Trough
  - **-0.5:** Rut (concave cylinder)
  - **-0.25:** Saddle Rut
  - **0.0:** Saddle (a perfect saddle shape, where `kappa_1 = -kappa_2`)
  - **+0.25:** Saddle Ridge
  - **+0.5:** Ridge (convex cylinder)
  - **+0.75:** Dome
  - **+1.0:** Spherical Cap (a perfectly convex bump)

---

### `curvedness_vv`
- **Type:** Float  
- **Description:**  
  **PREFERRED VERSION** - A measure of the overall magnitude of curvature at a given point on the surface, calculated as `sqrt((kappa_1^2 + kappa_2^2) / 2)` after vector voting. Always non-negative.

---

### `shape_index_cat`
- **Description:**  
  A numerical property that categorizes the continuous `shape_index_VV` value into one of nine shape classes, storing a single representative float value for each class. Based on the preferred vector-voted shape index.
- **Values:**
  - **-0.75:** Trough
  - **-0.5:** Rut (concave cylinder)
  - **-0.25:** Saddle Rut
  - **0.0:** Saddle (a perfect saddle shape, where `kappa_1 = -kappa_2`)
  - **+0.25:** Saddle Ridge
  - **+0.5:** Ridge (convex cylinder)
  - **+0.75:** Dome
  - **+1.0:** Spherical Cap (a perfectly convex bump)

---

### `Verticality`
- **Description:**  
  The angle of the triangle's surface plane relative to the XY plane of the tomogram, ranging from 0° (perfectly horizontal, parallel to XY) to 90° (perfectly vertical, perpendicular to XY). Calculated using the vector-voted normal (`n_v`).

---

### `self_dist_min`
- **Description:**  
  The shortest distance from the center of a triangle to another part of the same surface, measured along the triangle's normal vector. For each triangle, two rays are cast from its center along the normal vector (`n_v`), one in each direction. The first intersection point with the mesh is found in each direction; `self_dist_min` is the shorter of these distances. If no intersection is found, the value is NaN.

---

### `self_id_min`
- **Description:**  
  The index of the triangle on the same surface that was intersected at the distance reported in `self_dist_min`.

---

### `self_dist_far`
- **Description:**  
  The longest distance from the center of a triangle to another part of the same surface, measured along the triangle's normal vector. This value is calculated in the same process as `self_dist_min`.

---

### `self_id_far`
- **Description:**  
  The index of the triangle on the same surface that was intersected at the distance reported in `self_dist_far`.

---

### `average_width`
- **Description:**  
  The width of the overall connected component. This is not an average for individual triangles, but a measurement representing the connected region as a whole.

---

### `CLASS_dist`
- **Description:**  
  The shortest Euclidean distance from a given triangle to the closest triangle that belongs to a different class or category. `[CLASS]` is a placeholder for the name of the other surface being measured against.

---

### `CLASS_neighbor_index`
- **Description:**  
  The index of the nearest triangle on the target surface specified by `[CLASS]`.

---

### `CLASS_orientation`
- **Description:**  
  The relative angle between the normal vector (`n_v`) of a triangle on the current surface and the normal vector of its neighbor on the target surface specified by `[CLASS]`. The angle is calculated as the acute angle, always between 0 and 90 degrees.

---

### `Subcompartment`
- **Description:**  
  A label assigned to a group of triangles on a larger surface to define a region of biological interest.

---

### `xyz_x`, `xyz_y`, `xyz_z`
- **Description:**  
  The coordinates representing the center of each triangle in the mesh.

---

### `normal_x`, `normal_y`, `normal_z`
- **Description:**  
  The x, y, and z components of the normal vector representing the orientation of the surface at each point, as a unit vector in 3D space.

---

### `n_v_x`, `n_v_y`, `n_v_z`
- **Description:**  
  The x, y, and z components of the voted normal vector for each triangle.

---

### `t_v_x`, `t_v_y`, `t_v_z`
- **Description:**  
  The components of a tangent vector assigned to each vertex of the surface mesh.

---

### `t_1_x`, `t_1_y`, `t_1_z`
- **Description:**  
  The components of the first principal direction (tangent) vector at each vertex, typically aligned with the direction of maximum curvature on the surface at that point.

---

### `t_2_x`, `t_2_y`, `t_2_z`
- **Description:**  
  The components of the second principal direction (tangent) vector at each vertex, typically perpendicular to the first principal direction.

---

### `Average_width`
- **Description:**  
  A value representing the average width of an entire connected component of a surface. This is a per-component metric, not a per-triangle one.

---

### `Thickness`
- **Description:**  
  The per-triangle membrane bilayer thickness (the leaflet-to-leaflet separation), in the surface's distance units. It is measured (by `morphometrics measure_thickness`) by sampling the raw tomogram density along each triangle's normal vector and fitting the two leaflet density peaks with a dual Gaussian; the thickness is the distance between the fitted leaflet centers. Triangles where a bilayer cannot be resolved are reported as NaN. See also `bilayer_resolution`.

---

### `offset`
- **Description:**  
  The signed distance (along the triangle's normal) from the triangle center to the fitted bilayer midplane, from the same `measure_thickness` fit. A nonzero value means the mesh vertex sits slightly off the density-defined membrane center.

---

### `bilayer_resolution`
- **Description:**  
  A per-triangle reliability flag in the range [0, 1] for the reported `Thickness`: how clearly the two bilayer leaflets are resolved in the local density profile. A value near 1 means two cleanly separated leaflets; values toward 0 mean the leaflets are merged into a single peak and the thickness was recovered with help from the whole-surface average fit (and tends to read slightly thin). A practical cutoff is 0.5 — values ≥ 0.5 are high-confidence measurements and values below 0.5 are lower-confidence. NaN where no thickness was measured. See also `forced_bilayer_prior`.

---

### `forced_bilayer_prior`
- **Description:**  
  A per-triangle flag (1 or 0) for the reported `Thickness`. It is 1 where the thickness was obtained only because `thickness_measurements: force_bilayer_prior` was enabled: the whole-surface average profile did not resolve two leaflets, was fitted as a bilayer anyway, and that forced fit seeded the recovery of this triangle's merged profile. These are the least reliable thickness measurements: with the leaflets merged, the fitted separation trades off against leaflet width, so the value leans on the fit model more than on resolved density, and it shifts with `average_radius` and tomogram resolution. Note that on such data the triangles that *do* resolve strictly are not an unbiased reference either: they are selected for unusually wide leaflet separations and can read thick. Use flagged values for relative comparisons within matched data at most, and check them together with `bilayer_resolution`. It is 0 for every other triangle (including NaN-thickness triangles), and 0 everywhere when the option is off (the default).
