# Initial Research Run Report

**Project:** Mechanistic Interpretability of Spatial Reasoning in VLMs  
**Model:** Qwen/Qwen2-VL-2B-Instruct  
**Hardware:** Apple M3 Pro MacBook Pro, MPS backend, fp16  
**Date:** 18 May 2026  
**Scope:** Day 1 to Day 4 pilot experiments

## Executive Summary

The initial run is a strong success overall. The pilot established that Qwen2-VL has a robust, linearly accessible spatial representation, that spatial positions are encoded in deeper layers, and that relation-specific directions are distinct from raw position directions. The model also shows a language-prior bypass on some examples, and the harder follow-up experiments added useful nuance:

- Spatial structure remains recoverable even when surface accuracy drops on harder images.
- A causal direction exists, but activation patching shows a threshold effect rather than smooth control.
- Some binary directions are stimulus-specific, while the broader 4-way spatial structure generalizes much better.

The main scientific takeaway is that the model does not merely answer spatial questions from a single surface heuristic. It contains a real internal spatial subspace, but the decoding and causal control of that subspace are more complex than the clean probe results alone suggest.

## Overall Verdict

**GO, with refinement.**

The project is viable, and the first run already supports a coherent research direction. The right next move is not to restart the project, but to deepen the causal story: isolate the circuit that converts position information into relation judgments, explain why binary causal patching is thresholded, and test whether the same mechanism holds in more natural settings.

## What We Learned

### 1. The basic infrastructure works
The hook/activation pipeline is reliable. We can capture layer activations across Qwen2-VL during spatial inference and reuse the collector cleanly across repeated runs.

### 2. Spatial and non-spatial states are separable
A linear probe distinguishes spatial from non-spatial activations at 100% accuracy across the tested layers. This means the model builds a clearly identifiable spatial computation mode.

### 3. Positions are linearly encoded in deeper layers
Position encoding becomes strong in the middle-to-late layers, with peak results around layer 21.

Key result:
- Layer 21: R² = 0.876 for x, 0.981 for y in the original pilot
- On harder cluttered images: Layer 21 still reached R² = 0.755 average

This confirms that spatial location is not hidden in an opaque way. It is accessible to linear methods.

### 4. Relations are not the same as positions
The relation directions are almost orthogonal to raw position directions.

Most important pilot result:
- Above/below direction overlap with y-position direction = 0.037

That is the strongest mechanistic result so far. It argues that the model is not simply reading off coordinates. There is a separate relational subspace.

### 5. The model has a language-prior shortcut
The vision-bypass experiment showed that the model can answer from prior knowledge when the image conflicts with common sense or when the image is blank/shuffled.

Interpretation:
- There is a real visual pathway.
- There is also a prior-driven pathway.
- Some errors are likely caused by the prior pathway dominating.

### 6. Harder images reveal a decode gap, not a representational collapse
In Experiment 6, the model’s accuracy dropped on cluttered or adversarial scenes, but the internal probe stayed strong.

Key results:
- Level 3 cluttered accuracy: 60%
- Level 3 4-way probe accuracy: 100%
- Cluttered position R² at layer 21: 0.755

This is important: the representation survives, but the final answer can fail. That points to a downstream decoding or decision bottleneck rather than a disappearance of the spatial code.

### 7. Causal control exists, but it is thresholded
The activation patching experiment did not produce smooth flips at unit scale, but it did flip perfectly once the patch was amplified.

Key results:
- Targeted flip rate at 1x: 20%
- Control flip rate: 0%
- Flip rate at 1.5x and above: 100%

Interpretation:
- The learned direction is causally meaningful.
- The raw direction is not always sufficient at natural scale.
- The mechanism may be distributed, nonlinear, or require stronger intervention to overcome competing features.

### 8. Generalization depends on the task definition
Experiment 8 split the story into two parts:

- Binary above/below direction trained on red circle + blue square did **not** generalize well to novel shapes, swapped colors, or small objects.
- The 4-way relation classifier **did** generalize perfectly to novel shapes.

Key results:
- Novel-shape binary accuracy: 50%
- Small-object binary accuracy: 50%
- Three-object binary accuracy: 66.7%
- Swapped-color binary accuracy: 50%
- 4-way novel-shape accuracy: 100%

Interpretation:
- The binary direction is entangled with stimulus identity.
- The broader relational structure is much more invariant.
- The 4-way formulation is probably the better object for the full mechanistic story.

## Conclusions

The initial run supports four core claims:

1. Qwen2-VL contains a genuine spatial representation that is linearly accessible.
2. Spatial positions and spatial relations are represented in different subspaces.
3. The model sometimes uses a language-prior shortcut instead of the image.
4. The causal and generalization story is more subtle than the probe story, which is exactly where the interesting research is.

The strongest single headline is still the orthogonality result: relation directions are not just position directions. That is the anchor for the next phase.

## What This Means Scientifically

The project now has enough evidence to argue that the model likely computes spatial relations through an intermediate transformation rather than direct coordinate comparison. The next question is no longer whether a spatial representation exists. The next question is:

**What circuit converts position information into relation judgments, and under what conditions does that circuit fail or get bypassed?**

That is a meaningful mechanistic interpretability problem.

## Recommended Next Steps

### Immediate next steps
1. **Refine the causal experiment.**
   - Rework Exp 7 so the intervention is tested more systematically across scales and relation types.
   - Focus on a cleaner causal score rather than one aggregate flip threshold.

2. **Shift from binary to 4-way relation analysis.**
   - The 4-way classifier generalized better than the binary above/below direction.
   - This suggests the 4-way space may be the better target for circuit discovery.

3. **Probe the decode bottleneck.**
   - Compare internal relation linearly decodable states with final answer logits.
   - Identify where the representation is lost or overridden.

4. **Test on natural images.**
   - The synthetic stimuli establish the mechanism.
   - Natural scenes will show whether the same circuit survives outside the controlled setting.

### Medium-term next steps
5. **Train SAEs or feature dictionaries on layers 14-27.**
   - This should expose features corresponding to position, relation, and possibly prior-driven shortcuts.

6. **Map the attention heads and MLPs involved.**
   - Use activation patching, ablations, and head-level interventions to localize the position-to-relation transformation.

7. **Separate stimulus identity from relation geometry.**
   - The binary generalization failure suggests entanglement.
   - Try more diverse training sets or explicitly factorized stimuli.

### Longer-term next steps
8. **Build a causal story for the spatial circuit.**
   - Identify the representation.
   - Identify the transformation.
   - Identify the decision layer.
   - Show what happens when each is perturbed.

## Mentor-Ready Summary

If you need a short meeting version, use this:

> We now have strong evidence that Qwen2-VL contains a linearly decodable spatial subspace, that relation directions are nearly orthogonal to raw position directions, and that the model sometimes answers from language priors rather than vision. On harder images the representation survives while accuracy degrades, and causal patching shows thresholded control. The next step is to localize the circuit that turns position into relation judgments and test whether the 4-way relation subspace generalizes in more natural settings.

## Status

**Initial run complete.**  
**Project direction validated.**  
**Next phase: causal localization and generalization on more realistic data.**
