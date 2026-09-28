# Experiments and statistical analysis

## Experiment engine

The `taf.experiments` module defines an experiment declaratively and executes it without any user interface:

```python
from taf.experiments import ExperimentConfig, ExperimentType, run_experiment

config = ExperimentConfig(
    experiment_type=ExperimentType.DATASET_BENCHMARK,
    name="lsb-vs-fsvc-vctk",
    dataset_id="vctk",            # example | vctk | librispeech | all (library:<id> within the platform)
    file_limit=4,                 # dataset_path="C:/my/corpus" is also accepted
    methods=["LSB_METHOD", "FSVC_METHOD"],
    metrics=["SNR_METRIC", "PESQ_METRIC"],
    attacks=["additive_noise"],   # each attack is paired with a no-attack baseline
    payload_lengths=[16, 32],
    repetitions=3,
    random_seed=42,
)
run = run_experiment(config)      # per-row failures are recorded, never raised
print(run.status, run.summary["overall"])
run.to_csv("detailed_results.csv")
```

Nine experimental designs share one configuration schema. Each answers one research question about one of the four
properties of an information-hiding system:

| Property | Design | Question | Primary analysis |
| --- | --- | --- | --- |
| Imperceptibility | `perceptual_quality` | How much does embedding degrade the audio? | Per-metric ranks, mean rank, Friedman + Holm–Wilcoxon |
| Robustness | `attack_robustness` | Which method survives a battery of attacks? | Method × attack BER matrix, paired comparison per attack |
| Robustness | `robustness_curve` | At which attack strength does each method break down? | Dose–response curve of BER with bands, breakdown point |
| Capacity | `embedding_capacity` | What is the largest payload a method carries reliably? | Per-file capacity (bits, bit/s), bootstrap interval |
| Security | `detectability` | Can a steganalyser tell stego from cover? | Ensemble steganalysis, Wilson interval, binomial test |
| Multi-criteria | `tradeoff_curve` | How does embedding strength trade transparency for robustness? | Quality–BER curve over a method parameter, Pareto front |
| Multi-criteria | `method_comparison` | Which methods are not beaten on every criterion at once? | Pareto front, paired tests per criterion |
| Multi-criteria | `dataset_benchmark` | How do methods perform across a dataset? | Summary per method, clean and under attack |
| — | `research_experiment` | Any combination of factors (exploratory) | Every analysis that applies |

All designs except `detectability` execute a full factorial design (files × methods × payload lengths × repetitions ×
attack variants, each attack next to a no-attack baseline) and differ in the factors they fix and the analysis applied.
Their results are normalised rows: file, method with its parameters, payload in bits and bits per second, attack with
its resolved parameters, BER, bit accuracy, metric values, timings, status and failure kind. The `detectability` design
runs the [steganalysis procedure](steganalysis.md) for every method and payload length.

Methods are named by catalogue name or by a specification that sets constructor parameters, e.g.
`"QIM_METHOD:step_scale=0.1"`; the label and the parameters are recorded in every row. The two curve designs generate
their conditions from one parameter:

```python
from taf.experiments.sweeps import ParameterSweep

# robustness_curve: one attack parameter from mild to harsh, read in the given order
attack_sweep = ParameterSweep(target="awgn", parameter="snr_db", values=[40, 30, 20, 15, 10, 5, 0])
# tradeoff_curve: one method parameter, usually the one the catalogue marks as the embedding strength
method_sweep = ParameterSweep(target="QIM_METHOD", parameter="step_scale", values=[0.025, 0.05, 0.1, 0.2, 0.4])
```

The breakdown point of a robustness curve is where, in sweep order, the BER first exceeds the usable threshold (0.10),
linearly interpolated between the last usable and the first unusable setting; it is reported as `never` or `always`
when the curve does not cross.

## Experimental protocol

* **Randomness.** A single experiment seed determines every random quantity. When none is given, one is drawn and
  stored in the exported configuration. Messages are seeded by their length, so messages of different lengths are
  independent draws and adding a payload length leaves the others unchanged. Attack realisations are seeded by file,
  repetition and attack, but not by method: repetitions sample the channel independently, while every method meets
  the same noise realisation within a trial (common random numbers), which keeps comparisons between methods paired.
* **Blind extraction.** Decoding runs on a fresh method instance that has seen neither the cover nor the message.
* **Separation of effects.** Cover-to-stego metrics (imperceptibility) are computed once per embedded signal and are
  never mixed with stego-to-attacked metrics (attack damage). Multi-valued metrics are reported per named component.
* **Failures are not bit errors.** A trial that yields no decoded message is labelled `over_capacity`, `encode_error`,
  `io_error`, `attack_error` or `decode_error`, and has no BER. Bit-level statistics use completed trials and are
  reported together with the completion rate. Where one robustness figure must include failures, they are scored at
  chance level (BER = 0.5).

## Statistical analysis

The file is the unit of replication, because rows of one file (payloads, repetitions, attacks) are correlated.

* **Uncertainty.** 95% intervals come from a cluster bootstrap that resamples whole files.
* **Method comparison.** Methods are compared on per-file means in a paired design. Two methods are compared with the
  Wilcoxon signed-rank test; more than two with the Friedman test followed by Holm-corrected pairwise Wilcoxon tests,
  rank-biserial effect sizes and the Nemenyi critical difference (Demšar, 2006). Below a certain number of files, no
  Holm-corrected pairwise test can reach p < 0.05, whatever the data: 6 files for two methods, 7 for three and 15 for
  all 30. The preview warns when the dataset is smaller than that.
* **Multi-criteria summary.** Methods are summarised by their Pareto front over bit accuracy, robustness and every
  metric with a declared direction. A weighted score is computed only when weights are supplied, because it depends on
  the weights and, through normalisation, on the set of compared methods.
* **Capacity.** Capacity is determined per file as the largest tested payload below the first failing one, where a
  payload passes when all trials complete and the bit-accuracy and BER thresholds are met. It is reported in bits and in
  bits per second, and summarised across files with a bootstrap interval. Payloads of up to 8192 bits can be tested.

## Evaluation block

Next to the analysis of its design, every factorial run carries a shared evaluation block
(`taf.experiments.scenarios.evaluation`, `summary["evaluation"]`), computed with the same estimators:

* **Per method**, without and under attack: mean with its interval, median, SD, range, quartiles and IQR of BER over
  completed trials; Tukey box summaries; recovery rates at BER = 0, ≤ 1 % and ≤ 5 % (failed trials count as not
  recovered). With *L*-bit payloads BER moves in steps of 1/*L*, so below 100 bits "≤ 1 %" can only mean BER = 0; the
  block records this.
* **Per attack**: the BER increase it causes, as the mean over files of attacked minus clean BER (each file paired with
  itself), the methods ordered by their BER under it, and paired comparisons of the methods.
* **Per attack family observed at several strengths** (severity levels or a swept parameter): the degradation curve
  and a test of monotonic trend.
* **Quality** cover-vs-stego kept apart from attack damage stego-vs-attacked; **payload** (bits, bit/s) against BER and
  quality; **processing time** and real-time factor, compared between methods only when measured with one worker.
* **Correlations** (payload vs BER, quality and time; attack strength vs BER) as Spearman ρ on per-(file, level) means,
  with a permutation test that shuffles values *within* files, a bootstrap interval over files, and Holm correction
  over all correlation tests of the run.
* **Findings**: statements computed from these figures (largest BER increase with its interval, methods whose upper
  95 % bound of BER stays ≤ 0.01 under an attack, failures, between-file variability, significant trends, outcome of
  the paired tests with the largest significant effect, and when too few files make significance unreachable). They
  also open the LaTeX and Markdown reports.

Confidence level, bootstrap resamples and seed, tests, effect size, correction, permutations and seed are stored with
the block, so every figure can be recomputed from the stored trials.

## Provenance

Every run produces a manifest (`ExperimentRun.manifest`, `GET /api/runs/{id}/manifest.json`). It records the
package version and source commit (with a flag for uncommitted changes), the Python, platform, numerical-library and
FFmpeg versions, the SHA-256 digest, sampling rate and duration of every input file, the resolved seed and the
expanded attack list. It also states whether timings are comparable: timings are comparable only with
`max_workers = 1`, and speed is otherwise excluded from method comparisons.

## Reproduction of single trials and reports

With matching inputs, software and model weights, a stored trial can be re-synthesised from its run seed
(`taf.experiments.inspector.resynthesize`): cover, stego, attacked signal and residual (stego − cover), with the decoded
message checked against the stored one. `taf.experiments.reporting.build_report` summarises a run as a LaTeX (booktabs)
or Markdown report: an *Experimental setup* paragraph generated from the configuration and manifest, the result tables
with 95% intervals, and the statistical tests.
