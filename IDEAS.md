# Cool ideas to try to improve and extend CLT work

We welcome any contribution on these topics and are happy to collaborate. Please reach out if you are interested in working on any of these directions.

---

## 1. Reducing compute and model size

The main limitation of CLTs, beyond the heavy post-training engineering required for attribution graph analysis, is their size and computational cost.

It is therefore important to explore modifications to the current architecture that reduce FLOPs and number of parameters.

Ideas along this direction are very welcome. Any variant that can be naturally implemented within this library can be added.

---

## 2. Attribution graph complexity and L0

A second limitation is the analysis of the attribution graph, which becomes very slow and difficult when there are too many nodes.

We observed that it is often cheaper and more efficient to start with fewer nodes (i.e., lower L0 in the CLT), rather than relying on heavy attribution graph pruning.

This leads to a core challenge:
- How can we maintain high replacement scores (especially on the sentences of interest) while using as few nodes as possible?

Currently, we typically stop training when the average L0 per layer is between 3 and 10 features.  
How much lower can we go?

---

## 3. Improving replacement score

How can we further improve the replacement score?

Reconstruction error is only a proxy for replacement score. We have tried fine-tuning directly on replacement score at the end of training, but this approach is very noisy and did not lead to clear improvements.

There is likely a better way to optimize for this objective.

---

## 4. Training strategy

CLTs are large and expensive to train.

Should we always train them from scratch?  
Alternatively, we could:
- Fine-tune large pretrained CLTs  
- Use low-rank adaptation on smaller, more specific datasets  

The library will be improved to better support these workflows.

---

## 5. Number of features per layer

What should be the number of features per layer?

Currently, we use the same number of features per layer. However:
- Early layers seem to require more features  
- Later layers often end up with many dead features  

This suggests that a more flexible architecture, with a variable number of features per layer, could be beneficial.

---

## 7. Feature concentration in early layers

We observe that most features in CLTs — and even more so after attribution graph pruning — are located in layer 0.

While this can partly be explained by the cross-layer mapping in CLTs, which tends to favor early-layer neurons, it is unclear whether additional biases are at play.

Are there other sources of bias that we should try to remove?
