# Task: Implement `BiologicalFeatureEncoder`

Write a Python class named `BiologicalFeatureEncoder` that extends an existing `FrameMergingFeatureEncoder`. 

## Context
We are building a feature encoder for a PyTorch Autoencoder used for anomaly detection in video frames. The parent class (`FrameMergingFeatureEncoder`) squishes all objects in a frame into a single statistical vector (percentiles of speed, distance, etc.). 
We need to create a **Global-Local Feature Fusion**. The new `BiologicalFeatureEncoder` will inherit the global statistical features from the parent, but **enrich** the final `FeatureVector` by concatenating an additional array representing the exact physics and multi-hot semantic traits of the top N largest objects in the frame.

## Requirements

1. **Inheritance & Enrichment:**
   - Inherit from `FrameMergingFeatureEncoder`.
   - Override the `_encode_group(self, t: int, group: list[StateVector]) -> FeatureVector` method.
   - Inside the overridden method, call `super()._encode_group(t, group)` to get the base global statistical vector.
   - Concatenate the new object-slot features to this base vector and return the enriched `FeatureVector`.

2. **Top N Object Slots:**
   - Sort the `group` (list of `StateVector`) by `size` descending.
   - Take the top `MAX_OBJECTS = 5` this should be a parameter from configuration - default value is 5.
   - For each object, generate a 16-dimensional vector:
     `[is_present, x, y, vx, vy, size, trait_1, trait_2, ..., trait_10]`
   - `is_present` should be `1.0` for valid objects.
   - Pad any remaining slots (if the frame has fewer than 10 objects) with exactly `0.0` for all 16 dimensions.

3. **Multi-Hot Trait Mapping:**
   - Use the `state.type` (which corresponds to MS COCO dataset integer IDs) to generate 10 binary floats (`1.0` or `0.0`) based on physical/behavioral traits. 
   - Implement this mapping as a static method, module-level dictionary, or helper function. Here is the strict mapping logic to use:
     - `BIOLOGICAL` = {0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
     - `MOTORIZED` = {2, 3, 4, 5, 6, 7, 8}
     - `RIDEABLE` = {1, 2, 3, 4, 5, 6, 7, 8, 17, 20, 30, 31, 36, 37}
     - `HEAVY` = {2, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23, 57, 59, 60, 69, 72}
     - `PORTABLE` = {24, 25, 26, 27, 28, 29, 32, 33, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 63, 64, 65, 66, 67, 70, 73, 74, 75, 76, 77, 78, 79}
     - `ANCHORED` = {9, 10, 11, 12, 61, 71}
     - `AERIAL` = {4, 14, 29, 32, 33}
     - `FURNITURE` = {13, 56, 57, 58, 59, 60, 62, 68, 69, 72}
     - `WHEELED` = {1, 2, 3, 4, 5, 6, 7, 28, 36}
     - `HANDHELD` = {25, 34, 35, 38, 42, 43, 44, 64, 65, 67, 76, 78, 79}
   - This mapping should be configuration.

## Expected Output
Provide ONLY the production-ready Python code for `BiologicalFeatureEncoder` and other needed changes (probably only configuration)