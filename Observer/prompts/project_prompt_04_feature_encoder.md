Please add new FeatureEncoder called FrameMergingFeatureEncoder that will implement IFeatureEncoder

# How it should work
For input list[StateVector] group them by same time, and for every group prepare separate one FeatureVector.
FeatureVector will consist now as well some statistical data about position and speed of all objects in group.
Add only new implementation, and insert it inside supervisor. Do not any WEB part (server, or static html files). Pure adding of new implementation.
On output one FeatureVector is for one frame (multiple StateVector at same time).

## Proposed fields of FutureVector mapping. It should be calculated from (StateVector list for same time ) and then passed over to FeatureVector.vector:
    
    num_objects: float
    """Total number of tracked objects in the frame (K)."""

    dist_min: float
    """Minimum Euclidean distance between any pair of objects."""

    dist_p10: float
    """10th percentile of Euclidean distances between all object pairs."""

    dist_p50: float
    """Median (50th percentile) of Euclidean distances between all object pairs."""

    dist_p90: float
    """90th percentile of Euclidean distances between all object pairs."""

    rel_speed_p10: float
    """10th percentile of relative speeds ||v_i - v_j|| between all object pairs."""

    rel_speed_p50: float
    """Median relative speed ||v_i - v_j|| between all object pairs."""

    rel_speed_p90: float
    """90th percentile of relative speeds ||v_i - v_j|| between all object pairs."""

    rel_speed_max: float
    """Maximum relative speed between any pair of objects in the frame."""

    speed_p10: float
    """10th percentile of absolute object speeds ||v_i||."""

    speed_p50: float
    """Median absolute object speed ||v_i|| across all objects."""

    speed_p90: float
    """90th percentile of absolute object speeds ||v_i|| across all objects."""

    speed_max: float
    """Maximum absolute object speed ||v_i|| in the frame."""

    size_p10: float
    """10th percentile of object size"""

    size_p50: float
    """Median absolute object size"""

    size_p90: float
    """90th percentile of object size"""

    size_max: float
    """Maximum object size"""