import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .models import GraphData, InterventionData, InterventionResult
from ..config.settings import AppConfig

class DataLoader:
    """Handles loading and preprocessing of graph data."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Cached objects
        self._cached_data: Optional[GraphData] = None
        self._raw_graph_data = None

        # Intervention caches
        self._intervention_data = None
        self._processed_intervention_data = None

    def load_attribution_graph(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        list,
    ]:
        """Load attribution graph from saved file."""

        if self._raw_graph_data is None:
            self._raw_graph_data = torch.load(
                self.config.attr_graph_path,
                map_location="cpu",
            )

        data = self._raw_graph_data

        feature_mask = data["feature_mask"].numpy()
        adjacency = data["sparse_pruned_adj"].T.numpy()
        feature_indices = data["feature_indices"].numpy()

        input_tokens = data["input_tokens"].numpy()
        input_str = data["input_string"]
        token_strings = data["token_string"]
        n_layers = data["n_layers"]

        logit_tokens = data.get("logit_tokens", None)
        logit_probabilities = data.get("logit_probabilities", None)

        top_logit_token_idx = logit_probabilities.argmax().item()
        top_logit_token = logit_tokens[top_logit_token_idx].item()

        topk_logit_strings = data["logit_token_strings"]
        topk_logit_probs = logit_probabilities

        feature_list_intersection = data.get("feature_list_intersection", None)

        intervention_top_tokens = None
        possible_keys = [
            "intervention_top_tokens",
            "intervention_data",
            "interventions",
            "intervention_results",
        ]

        for key in possible_keys:
            if key in data:
                intervention_top_tokens = data[key]
                break

        self._intervention_data = intervention_top_tokens

        return (
            adjacency,
            feature_indices,
            input_tokens,
            input_str,
            token_strings,
            top_logit_token,
            topk_logit_strings,
            topk_logit_probs,
            feature_list_intersection,
            feature_mask,
            n_layers
        )

    def load_feature_dict(
        self,
        layer: int,
        feat_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Load feature dictionary for a specific layer and feature."""

        dict_path = (
            Path(self.config.dict_base_folder)
            / f"layer{layer}"
            / f"feature_{feat_idx}_complete.json"
        )

        if not dict_path.exists():
            return None

        try:
            with open(dict_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def preprocess_data(self) -> GraphData:
        """Load and preprocess all graph data."""

        if self._cached_data is not None:
            return self._cached_data

        (
            adjacency,
            feature_indices,
            token_ids,
            input_str,
            token_strings,
            top_logit_token,
            topk_logit_strings,
            topk_logit_probs,
            feature_list_intersection,
            feature_mask,
            n_layers
        ) = self.load_attribution_graph()

        n_logits = len(topk_logit_strings)

        print("n_logits:", topk_logit_strings)
        print("logit probs:", topk_logit_probs)

        n_tokens = len(token_strings)
        n_features = feature_indices.shape[0]
        n_errors = n_tokens * n_layers

        print(f"number of tokens: {n_tokens} ({token_ids})")
        print("n_features:", n_features)
        print("n_errors:", n_errors)
        print("n_logits:", n_logits)

        expected = n_features + n_tokens + n_errors + n_logits

        assert (
            adjacency.shape[0] == expected
        ), f"Adjacency matrix row count {adjacency.shape[0]} does not match expected {expected}"

        adjacency_from_embeddings = np.concatenate(
            [
                adjacency[n_features + n_errors : n_features + n_errors + n_tokens, :n_features],
                adjacency[n_features + n_errors : n_features + n_errors + n_tokens, [-n_logits]],
            ],
            axis=1,
        )

        adjacency = np.concatenate(
            [adjacency[:, :n_features], adjacency[:, [-n_logits]]],
            axis=1,
        )

        adjacency = adjacency[:n_features]

        edge_to_logit_mask = adjacency[:, -1] > 0
        feature_mask = feature_mask[:n_features] & edge_to_logit_mask

        print(
            "Number of features in the mask",
            torch.tensor(feature_mask).float().sum().item(),
        )

        frequency_path = (
            Path(self.config.dict_base_folder).parent
            / "feature_frequency.json"
        )

        frequency_mask = np.ones(n_features, dtype=bool)

        if frequency_path.exists():
            try:
                with open(frequency_path, "r") as f:
                    frequency_data = json.load(f)

                for i, (ctx_pos, layer, feat_idx) in enumerate(feature_indices):
                    layer_str = str(layer)

                    if layer_str in frequency_data and feat_idx < len(
                        frequency_data[layer_str]
                    ):
                        frequency = frequency_data[layer_str][feat_idx]

                        if frequency > self.config.high_frequency_pruning_threshold:
                            frequency_mask[i] = False
                            print(
                                f"Frequency filtering removed layer {layer} feature {feat_idx}"
                            )

                print(
                    f"Frequency filtering removed {(~frequency_mask).sum()} features"
                )

            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Could not load frequency data: {e}")

        feature_mask = feature_mask & frequency_mask
        feature_indices = feature_indices[feature_mask]

        active_feature_frequencies: Dict[Tuple[int, int, int], float] = {}

        if frequency_path.exists():
            try:
                with open(frequency_path, "r") as f:
                    frequency_data = json.load(f)

                for ctx_pos, layer, feat_idx in feature_indices:
                    layer_str = str(layer)

                    if layer_str in frequency_data and feat_idx < len(
                        frequency_data[layer_str]
                    ):
                        frequency = frequency_data[layer_str][feat_idx]

                        active_feature_frequencies[
                            (int(ctx_pos), int(layer), int(feat_idx))
                        ] = frequency

            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(
                    f"Could not load frequency data for active features: {e}"
                )

        print(
            "Number of features in the graph after frequency pruning:",
            len(feature_indices),
        )

        adjacency = adjacency[feature_mask][:, np.append(feature_mask, [True])]

        adjacency_from_embeddings = adjacency_from_embeddings[
            1:, np.append(feature_mask, [True])
        ]

        prompt_length = len(token_strings)

        self._cached_data = GraphData(
            nodes=[],
            edges=[],
            adjacency_matrix=adjacency,
            active_mask=feature_mask,
            feature_indices=feature_indices,
            input_tokens=token_strings,
            input_str=input_str,
            n_layers=n_layers,
            prompt_length=prompt_length,
            token_x_positions=[],
            top_logit_token=top_logit_token,
            top5_logit_tokens=topk_logit_strings[:5],
            top5_logit_probs=topk_logit_probs[:5],
            feature_list_intersection=feature_list_intersection,
            feature_frequencies=active_feature_frequencies,
            embedding_adjacency=adjacency_from_embeddings,
        )

        print(
            "GraphData created with embedding_adjacency shape:",
            adjacency_from_embeddings.shape
            if adjacency_from_embeddings is not None
            else None,
        )

        return self._cached_data

    def clear_cache(self):
        """Clear cached data to force reload."""

        self._cached_data = None
        self._raw_graph_data = None
        self._intervention_data = None
        self._processed_intervention_data = None

    def get_processed_intervention_data(
        self,
    ) -> Optional[List[Optional[InterventionData]]]:
        """Process intervention data."""

        if self._intervention_data is None:
            return None

        if self._processed_intervention_data is not None:
            return self._processed_intervention_data

        processed_data = []

        for intervention_item in self._intervention_data:

            if intervention_item is None:
                processed_data.append(None)
                continue

            if "feature_info" in intervention_item and "interventions" in intervention_item:

                intervention_results = []

                for interv in intervention_item["interventions"]:
                    intervention_results.append(
                        InterventionResult(
                            intervention_value=interv["intervention_value"],
                            tokens=interv["tokens"],
                            probabilities=interv["probabilities"],
                            baseline_token=interv["baseline_token"],
                            baseline_prob_original=interv["baseline_prob_original"],
                            baseline_prob_after_intervention=interv[
                                "baseline_prob_after_intervention"
                            ],
                            baseline_prob_change=interv["baseline_prob_change"],
                        )
                    )

                processed_data.append(
                    InterventionData(
                        feature_info=intervention_item["feature_info"],
                        interventions=intervention_results,
                    )
                )

            else:

                intervention_results = [
                    InterventionResult(
                        intervention_value=-10.0,
                        tokens=intervention_item["tokens"],
                        probabilities=intervention_item["probabilities"],
                        baseline_token=intervention_item["baseline_token"],
                        baseline_prob_original=intervention_item[
                            "baseline_prob_original"
                        ],
                        baseline_prob_after_intervention=intervention_item[
                            "baseline_prob_after_intervention"
                        ],
                        baseline_prob_change=intervention_item[
                            "baseline_prob_change"
                        ],
                    )
                ]

                processed_data.append(
                    InterventionData(
                        feature_info={
                            "layer": -1,
                            "position": -1,
                            "feature_idx": -1,
                        },
                        interventions=intervention_results,
                    )
                )

        self._processed_intervention_data = processed_data
        return processed_data

    # def cluster_features(self, n_clusters: int = 5, use_activation_rate: bool = True) -> Optional[np.ndarray]:
    #     """Cluster features based on correlation matrices."""
        
    #     correlation_path = Path(self.config.attr_graph_path).parent / "feature_correlation_matrices.pt"
    #     if not correlation_path.exists():
    #         return None
        
    #     try:
    #         from sklearn.cluster import KMeans
            
    #         data = torch.load(correlation_path, map_location='cpu')

    #         if use_activation_rate:
    #             correlation_matrix = data['activation_rate_matrix']
    #         else:
    #             correlation_matrix = data['normalized_score_matrix']
            
    #         if correlation_matrix.is_sparse:
    #             correlation_matrix = correlation_matrix.to_dense()
            
    #         correlation_matrix = correlation_matrix.numpy()
            
    #         # Apply the same active mask filtering as used in the main graph
    #         # This ensures clustering only operates on active features
    #         graph_data = self.preprocess_data()
    #         active_mask = graph_data.active_mask
            
            
    #         # Apply active mask to correlation matrix (same as adjacency filtering)
    #         # Remove the last column if matrix is not square (remove extra dimension)
    #         if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
    #             correlation_matrix = correlation_matrix[:, :-1]
            
    #         # Apply active mask filtering
    #         correlation_matrix = correlation_matrix[active_mask][:, active_mask]
            
    #         # Ensure we have enough features to cluster
    #         if correlation_matrix.shape[0] < n_clusters:
    #             n_clusters = max(2, correlation_matrix.shape[0] // 2)
            
    #         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    #         cluster_labels = kmeans.fit_predict(correlation_matrix)
            
    #         return cluster_labels
            
    #     except ImportError:
    #         return None
    #     except Exception:
    #         return None

    # def compute_embeddings(self, use_activation_rate: bool = True) -> Optional[Dict[str, Any]]:
    #     """Compute UMAP/t-SNE embedding based on symmetric correlation matrices."""
        
    #     correlation_path = Path(self.config.attr_graph_path).parent / "feature_correlation_matrices.pt"
    #     if not correlation_path.exists():
    #         return None
        
    #     try:
    #         # Import dimensional reduction libraries
    #         try:
    #             from sklearn.manifold import TSNE
    #             from umap import UMAP
    #             tsne_available = True
    #             umap_available = True
    #         except ImportError:
    #             try:
    #                 from sklearn.manifold import TSNE
    #                 tsne_available = True
    #                 umap_available = False
    #             except ImportError:
    #                 return None
            
    #         data = torch.load(correlation_path, map_location='cpu')

    #         if use_activation_rate:
    #             correlation_matrix = data['activation_rate_matrix']
    #         else:
    #             correlation_matrix = data['normalized_score_matrix']
            
            
    #         if correlation_matrix.is_sparse:
    #             correlation_matrix = correlation_matrix.to_dense()
            
    #         correlation_matrix = correlation_matrix.numpy()
            
    #         # Remove the last column if matrix is not square (remove extra dimension)
    #         if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
    #             correlation_matrix = correlation_matrix[:, :-1]
            
    #         # Apply the same active mask filtering as used in the main graph
    #         # Load the adjacency matrix to get the active mask
    #         graph_data = self.preprocess_data()
    #         active_mask = graph_data.active_mask
            
    #         # Apply active mask to correlation matrix (same as adjacency filtering)
    #         correlation_matrix = correlation_matrix[active_mask][:, active_mask]
            
    #         # Matrix properties calculated for embedding
            
    #         # Make the matrix symmetric by averaging with its transpose
    #         symmetric_matrix = (correlation_matrix + correlation_matrix.T) / 2
            
    #         # Ensure diagonal is 1 (self-correlation)
    #         np.fill_diagonal(symmetric_matrix, 1.0)
            
    #         # Convert correlation to distance - handle negative values properly
    #         if use_activation_rate:
    #             # For activation rate (0-1 bounded), use simple distance
    #             distance_matrix = 1 - np.abs(symmetric_matrix)
    #         else:
    #             # For normalized score (can be negative), normalize to [0,1] first
    #             min_val = symmetric_matrix.min()
    #             max_val = symmetric_matrix.max()
    #             if max_val > min_val:
    #                 # Normalize to [0,1]
    #                 normalized_matrix = (symmetric_matrix - min_val) / (max_val - min_val)
    #                 # Convert to distance
    #                 distance_matrix = 1 - normalized_matrix
    #             else:
    #                 # All values are the same, use uniform distances
    #                 distance_matrix = np.ones_like(symmetric_matrix) * 0.5
    #                 np.fill_diagonal(distance_matrix, 0.0)
            
    #         # Ensure distances are non-negative and bounded
    #         distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
            
    #         # Compute embeddings
    #         results = {}
            
    #         # t-SNE embedding
    #         if tsne_available:
    #             perplexity_val = min(30, len(symmetric_matrix) - 1)
    #             tsne = TSNE(n_components=2, random_state=42, metric='precomputed', 
    #                        perplexity=perplexity_val, init='random')
    #             tsne_embedding = tsne.fit_transform(distance_matrix)
    #             results['tsne'] = tsne_embedding
            
    #         # UMAP embedding
    #         if umap_available:
    #             umap_reducer = UMAP(n_components=2, random_state=42, metric='precomputed')
    #             umap_embedding = umap_reducer.fit_transform(distance_matrix)
    #             results['umap'] = umap_embedding
            
    #         # Store additional info
    #         results['symmetric_matrix'] = symmetric_matrix
    #         results['distance_matrix'] = distance_matrix
    #         results['method_used'] = 'activation_rate' if use_activation_rate else 'normalized_score'
    #         results['active_indices'] = np.where(active_mask)[0]  # Store the mapping from filtered to original indices
            
    #         return results
            
    #     except Exception:
    #         return None
