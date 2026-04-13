from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np

@dataclass
class FeatureNode:
    """Represents a feature node in the graph."""
    id: int
    x: float
    y: float
    layer: Union[int, str]
    pos: Union[int, str]
    feature_idx: Union[int, str]
    token: str
    description: str
    config: Optional[Dict[str, Any]]
    node_id_original: int
    frequency: Optional[float] = None

@dataclass
class FeatureEdge:
    """Represents an edge between features."""
    from_node: int
    to_node: int
    weight: float

@dataclass
class InterventionResult:
    """Single intervention result for a specific value."""
    intervention_value: float
    tokens: List[str]
    probabilities: List[float]
    baseline_token: str
    baseline_prob_original: float
    baseline_prob_after_intervention: float
    baseline_prob_change: float

@dataclass
class InterventionData:
    """Data structure for intervention analysis with multiple intervention values."""
    feature_info: Dict[str, Any]  # layer, position, feature_idx
    interventions: List[InterventionResult]

@dataclass
class GraphData:
    """Complete graph data structure."""
    nodes: List[FeatureNode]
    edges: List[FeatureEdge]
    active_mask: np.ndarray
    adjacency_matrix: np.ndarray
    feature_indices: np.ndarray
    input_tokens: List[str]
    input_str: str
    n_layers: int
    prompt_length: int
    token_x_positions: List[float]
    top_logit_token: str
    top5_logit_tokens: Optional[List[str]] = None
    top5_logit_probs: Optional[np.ndarray] = None
    intervention_data: Optional[List[Optional[InterventionData]]] = None
    feature_list_intersection: Optional[List[Tuple[int, int, int]]] = None
    feature_frequencies: Optional[Dict[Tuple[int, int, int], float]] = None
    embedding_adjacency: Optional[np.ndarray] = None  # Edges from embedding tokens to features
