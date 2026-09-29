---
name: taf-audio-research
description: Design and review The A-Files audio steganography and watermarking research, including DSP, metrics, experiments, statistics and scientific writing.
---

# Audio research

Work with the rigor of a researcher in digital audio, speech processing, steganography and watermarking. Read repository steering and relevant implementation before changing scientific behavior.

1. Define the research question, measured property, baseline, factors, outcomes and scope of the claim. Keep imperceptibility, robustness, capacity, recovery and detectability distinct.
2. Verify equations and applicability against original papers or authoritative specifications. Cite verified sources in scientific documentation. Label adaptations and approximations; an algorithm name does not establish equivalence with a publication.
3. Specify sample rate, channels, amplitude range, dtype, framing, padding, synchronization, payload units and decoder side information. Check silence, quiet speech, short clips, clipping and numerical stability. Do not normalize away attacks or silently resample to make a metric succeed.
4. Use existing seeded-trial and manifest mechanisms. Record dataset selection, parameters, available file hashes, code/environment and model versions. Separate embedding distortion (cover vs. stego) from attack damage and post-attack recovery.
5. Use file-level replication and existing estimators in `src/taf/experiments/analysis.py`. Account for speaker/corpus dependence when the design requires it. Preserve pairing and appropriate multiple-comparison correction. Prevent train/test leakage in detectability studies. Report effect sizes, uncertainty, counts, exclusions and failure rates alongside significance.
6. Validate using relevant tests and representative audio. Describe actual evidence and its limits; bundled smoke samples do not establish population-level performance.

Failed decoding is not automatically BER 1.0: preserve missing values and `failure_kind`, and report recovery separately. Use metric metadata for directions and components. Distinguish bits, bits/sample and bits/second. Objective quality scores are proxies, not subjective listening evidence. Empirical detectability is conditional on the detector and protocol, not a proof of secrecy.

Write scientific prose with explicit hypotheses, methods, results and limitations. Separate observations from explanations. Keep cross-design analyses in the existing scenario/analysis pipeline and trace findings to measured data.
