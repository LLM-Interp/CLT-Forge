"""
Edge cache system for efficient edge lookups and interactions.
Precomputes all edge data once to avoid repeated adjacency matrix traversals.
"""

from typing import Dict, Set, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .models import FeatureNode

@dataclass
class EdgeData:
    """Single edge data structure."""
    from_node_id: int
    to_node_id: int
    from_node: FeatureNode
    to_node: FeatureNode
    weight: float
    edge_type: str  # "feature_to_feature", "feature_to_logit"
    
class EdgeCache:
    """
    Efficient edge cache that precomputes all edge data once.
    Provides fast lookups for edge interactions without matrix traversals.
    """
    
    def __init__(self):
        self._edges: List[EdgeData] = []
        self._node_to_edges: Dict[int, Set[int]] = {}  # node_id -> set of edge_indices
        self._edge_lookup: Dict[Tuple[int, int], int] = {}  # (from, to) -> edge_index
        self._adjacency_shape: Tuple[int, int] = (0, 0)
        self._is_cached = False
        
    def build_cache(self, nodes: List[FeatureNode], adjacency_matrix: np.ndarray):
        """Build the edge cache from nodes and adjacency matrix."""
        # Build edge cache for efficient lookups
        
        self._edges.clear()
        self._node_to_edges.clear()  
        self._edge_lookup.clear()
        self._adjacency_shape = adjacency_matrix.shape
        
        # Initialize node-to-edges mapping
        for i in range(len(nodes)):
            self._node_to_edges[i] = set()
        
        edge_count = 0

        # 1. Feature-to-feature edges (exclude logit AND embedding nodes)
        for i, node_i in enumerate(nodes[:-1]):  # Exclude logit node
            # Skip embedding nodes
            if hasattr(node_i, 'layer') and node_i.layer == "embedding":
                continue

            for j, node_j in enumerate(nodes[:-1]):  # Exclude logit node
                # Skip embedding nodes
                if hasattr(node_j, 'layer') and node_j.layer == "embedding":
                    continue

                if i != j and (node_i.node_id_original < adjacency_matrix.shape[0] and
                              node_j.node_id_original < adjacency_matrix.shape[1]):

                    weight = adjacency_matrix[node_i.node_id_original, node_j.node_id_original]
                    if weight != 0.0:
                        edge_data = EdgeData(
                            from_node_id=i,
                            to_node_id=j,
                            from_node=node_i,
                            to_node=node_j,
                            weight=weight,
                            edge_type="feature_to_feature"
                        )
                        
                        edge_idx = len(self._edges)
                        self._edges.append(edge_data)
                        self._edge_lookup[(i, j)] = edge_idx
                        self._node_to_edges[i].add(edge_idx)
                        self._node_to_edges[j].add(edge_idx)
                        edge_count += 1
        
        # 2. Feature-to-logit edges (exclude embedding nodes)
        if len(nodes) > 0:
            logit_node = nodes[-1]
            logit_node_id = len(nodes) - 1
            logit_col_idx = adjacency_matrix.shape[1] - 1

            for i, node_i in enumerate(nodes[:-1]):  # Exclude logit node itself
                # Skip embedding nodes
                if hasattr(node_i, 'layer') and node_i.layer == "embedding":
                    continue

                if (node_i.node_id_original < adjacency_matrix.shape[0] and
                    adjacency_matrix[node_i.node_id_original, logit_col_idx] != 0):
                    
                    weight = adjacency_matrix[node_i.node_id_original, logit_col_idx]
                    edge_data = EdgeData(
                        from_node_id=i,
                        to_node_id=logit_node_id,
                        from_node=node_i,
                        to_node=logit_node,
                        weight=weight,
                        edge_type="feature_to_logit"
                    )
                    
                    edge_idx = len(self._edges)
                    self._edges.append(edge_data)
                    self._edge_lookup[(i, logit_node_id)] = edge_idx
                    self._node_to_edges[i].add(edge_idx)
                    self._node_to_edges[logit_node_id].add(edge_idx)
                    edge_count += 1
        
        self._is_cached = True
        # Edge cache built successfully
    
    def get_edges_for_node(self, node_id: int) -> List[EdgeData]:
        """Get all edges connected to a specific node (fast O(1) lookup)."""
        if not self._is_cached or node_id not in self._node_to_edges:
            return []
        
        edge_indices = self._node_to_edges[node_id]
        return [self._edges[idx] for idx in edge_indices]
    
    def get_incoming_edges(self, node_id: int) -> List[EdgeData]:
        """Get incoming edges to a specific node (fast lookup)."""
        edges = self.get_edges_for_node(node_id)
        return [edge for edge in edges if edge.to_node_id == node_id]
    
    def get_outgoing_edges(self, node_id: int) -> List[EdgeData]:
        """Get outgoing edges from a specific node (fast lookup)."""
        edges = self.get_edges_for_node(node_id)
        return [edge for edge in edges if edge.from_node_id == node_id]
    
    def get_edge_by_nodes(self, from_node_id: int, to_node_id: int) -> Optional[EdgeData]:
        """Get edge between two specific nodes (fast O(1) lookup)."""
        edge_key = (from_node_id, to_node_id)
        if edge_key in self._edge_lookup:
            edge_idx = self._edge_lookup[edge_key]
            return self._edges[edge_idx]
        return None
    
    def get_all_edges(self) -> List[EdgeData]:
        """Get all cached edges."""
        return self._edges.copy()
    
    def get_highlighted_edge_sets(self, selected_node_id: Optional[int]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Get incoming and outgoing edge sets for highlighting (replaces slow matrix traversal).
        Returns (incoming_edge_keys, outgoing_edge_keys).
        """
        incoming_keys = set()
        outgoing_keys = set()
        
        if selected_node_id is None:
            return incoming_keys, outgoing_keys
        
        # Use cached edge data instead of matrix traversal
        incoming_edges = self.get_incoming_edges(selected_node_id)
        outgoing_edges = self.get_outgoing_edges(selected_node_id)
        
        for edge in incoming_edges:
            incoming_keys.add((edge.from_node_id, edge.to_node_id))
        
        for edge in outgoing_edges:
            if edge.edge_type == "feature_to_logit":
                outgoing_keys.add((edge.from_node_id, "logit"))
            else:
                outgoing_keys.add((edge.from_node_id, edge.to_node_id))
        
        return incoming_keys, outgoing_keys
    
    def is_cached(self) -> bool:
        """Check if cache has been built."""
        return self._is_cached
    
    def clear_cache(self):
        """Clear the edge cache."""
        self._edges.clear()
        self._node_to_edges.clear()
        self._edge_lookup.clear()
        self._is_cached = False
        print("🧹 Edge cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics for debugging."""
        return {
            'total_edges': len(self._edges),
            'feature_to_feature': len([e for e in self._edges if e.edge_type == 'feature_to_feature']),
            'feature_to_logit': len([e for e in self._edges if e.edge_type == 'feature_to_logit']),
            'nodes_with_edges': len(self._node_to_edges),
            'is_cached': self._is_cached,
            'adjacency_shape': self._adjacency_shape
        }
