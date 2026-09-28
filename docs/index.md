# The A-Files

**Reproducible evaluation of audio steganography and watermarking.**

The A-Files combines embedding methods, controlled channel distortions and objective audio measures
in one experiment engine. Evaluate payload recovery, perceptual distortion, empirical capacity,
robustness and statistical detectability on the same recordings and conditions.

| Components | Documentation |
| --- | --- |
| 27 embedding methods | [Mechanisms, implementation scope and sources](methods.md) |
| 26 attack classes, four codec shortcuts, five pipelines | [Transformations and severity controls](attacks.md) |
| 25 objective metrics | [Definitions, directions and interpretation](metrics.md) |
| Browser research platform | [Create, run, inspect and export experiments](ui.md) |

## Start an experiment

1. [Install the package](installation.md) and run the bundled decoding workflow.
2. [Start the research UI](ui.md) to define a protocol and select data and conditions.
3. Review the execution plan, run the experiment and inspect failures alongside successful trials.
4. Export the configuration, provenance, result tables and research report.

## Scientific interpretation

Each trial embeds a binary payload, optionally applies an attack and extracts with a fresh decoder.
Quality scores separate cover-to-stego distortion from stego-to-attacked damage. BER measures payload
recovery; a held-out steganalyser estimates detectability. These quantities answer different questions.

Read the [protocol](protocol.md), [statistical analysis](experiments.md) and [measurement definitions](research-capabilities.md)
before comparing methods. Citations identify scientific sources; local adaptations and optional pretrained
models are identified in the catalogues. Results reported in publications remain separate from local measurements.

![Evaluation architecture](functions.svg)

This static site documents the toolkit. The research application runs separately with a Python API,
PostgreSQL and the web client. [Polski przewodnik](README.pl.md) · [Project and licence](project.md).
