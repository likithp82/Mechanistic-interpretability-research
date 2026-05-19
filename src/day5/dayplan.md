# Day 5 Execution Plan

**Project:** Mechanistic Interpretability of Spatial Reasoning in Qwen2-VL  
**Date:** 18 May 2026  
**Goal:** Move from representational evidence to causal localization and failure analysis.

## Day 5 Objective

By the end of Day 5, the project should answer three questions:

1. Where is the spatial relation signal most causally useful?
2. When does the model have the right relation internally but still output the wrong answer?
3. Does the relation circuit transfer beyond synthetic stimuli and resist language-prior shortcuts?

## Run Order

### 1. Experiment 9: 4-Way Causal Intervention

**Purpose:** Test whether the 4-way relation subspace from Day 4 is causally sufficient to steer predictions.

**What to do:**
- Use the stable 4-way relation classifier as the causal target.
- Intervene on the relation direction for each class: above, below, left, right.
- Measure whether predictions shift toward the intended class.
- Compare against random-direction controls.

**Success criteria:**
- Targeted intervention improves the intended class prediction by at least 30 percentage points over control.
- Random-direction control stays near chance.
- The intervention works on at least 3 of the 4 relations.

**Why this first:**
- It is the cleanest next causal test after Exp 8.
- It is more stable than the binary above/below direction.

---

### 2. Experiment 10: Layer-Wise Causal Sweep

**Purpose:** Find the earliest layer where the relation subspace becomes causally effective, not just linearly decodable.

**What to do:**
- Repeat the causal intervention at layers 14, 21, and 27.
- Keep the same prompts and stimuli.
- Measure flip rate or class steering at each layer.
- Compare the causal strength to the probe strength.

**Success criteria:**
- At least one layer shows a clearly stronger causal effect than the others.
- The best layer should align with or follow the strongest probe layer from Day 2/3.
- Random control remains low at every layer.

**Why this second:**
- It localizes where the spatial relation computation becomes actionable.
- It helps separate representation from implementation.

---

### 3. Experiment 11: Decode Bottleneck Test

**Purpose:** Test whether the model has the correct relation internally but fails at the final answer stage.

**What to do:**
- Compare internal relation predictions with final output logits.
- Track cases where the probe says the correct relation but the model answers incorrectly.
- Compare hard images vs easy images.
- Compare congruent vs incongruent / prior-driven cases.

**Success criteria:**
- Clear evidence of a representational-to-output gap on harder or prior-conflicted cases.
- At least some examples where internal decoding is correct but the final answer is wrong.
- The gap is larger on cluttered/adversarial stimuli than on clean synthetic stimuli.

**Why this third:**
- Exp 6 already suggested a decode bottleneck.
- This is the best way to explain why behavior degrades while the representation remains strong.

---

### 4. Experiment 12: Head-Level Localization

**Purpose:** Identify which attention heads or MLP blocks implement the position-to-relation transformation.

**What to do:**
- Start from the best causal layer identified above.
- Test attention-head ablations, patching, or targeted interventions.
- Rank heads by impact on relation prediction.
- Check whether specific heads are specialized for vertical vs horizontal relations.

**Success criteria:**
- A small subset of heads or blocks explains a disproportionate share of the causal effect.
- Ablating those components reduces relation accuracy or intervention success.
- The result is interpretable enough to motivate a circuit diagram.

**Why this fourth:**
- It is the next real mechanistic step after layer localization.
- It turns the layer-level story into a circuit-level story.

## Optional Follow-Ups If Time Permits

### 5. Experiment 13: Harder Stimulus Robustness

**Purpose:** Stress-test the causal circuit on more varied clutter, occlusion, size changes, and adversarial layouts.

**Success criteria:**
- The circuit survives at least some realistic perturbations.
- Failures can be tied to a specific stimulus factor.

### 6. Experiment 14: Natural-Image Transfer

**Purpose:** Test whether the same relation circuit appears in real-world scenes, not only synthetic shapes.

**Success criteria:**
- At least partial transfer to natural images.
- Same or similar layer responds to spatial relations in real scenes.

### 7. Experiment 15: Prior-vs-Vision Conflict on the Refined Setup

**Purpose:** Test whether language priors still override the relation circuit once the causal target is refined.

**Success criteria:**
- Priors still create measurable interference.
- The refined circuit can be separated from prior-driven answers.

## Day 5 Output Criteria

By the end of the day, you should have:

- One refined causal experiment showing whether the 4-way spatial subspace is interventionally useful.
- One layer-localization result showing where causal effect is strongest.
- One decode-gap analysis showing whether failures come from output decoding rather than representation loss.
- A shortlist of candidate heads or blocks for the next round of circuit work.

## Recommended Working Rule

Do not expand scope until the first three experiments are complete.
If the causal intervention is weak, refine the target before moving to head-level localization.
If the causal intervention is strong, move immediately to layer and head localization.

## Minimal Success for Day 5

Day 5 is successful if it produces:

1. A causal result on the 4-way relation subspace.
2. A clear best layer for that causal effect.
3. Evidence for or against a decode bottleneck.

That would be enough to justify Day 6 circuit localization and natural-image validation.
