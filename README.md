# Overcoming the Representational Limitations of Amplitude Encoding for Quantum Neural Networks

This repository is the reproducibility package for the ICLR 2027 submission **“Overcoming the Representational Limitations of Amplitude Encoding for Quantum Neural Networks.”** It contains the quantum and classical models, preprocessing pipeline, benchmark experiments, statistical comparisons, and visualization scripts used to study how classical data encoding affects the representational power of quantum neural networks (QNNs).

## Paper summary

Amplitude encoding represents a $d$-dimensional feature vector with only $\mathcal{O}(\log_2 d)$ qubits, but state normalization introduces a fundamental limitation: conventional amplitude encoding is invariant to every nonzero real rescaling of an input. Inputs $x$ and $c x$, for $c \neq 0$, therefore produce the same encoded quantum state up to an unobservable global phase. Because this ambiguity occurs before the variational circuit, no downstream quantum architecture can distinguish such inputs.

The paper establishes this normalization-induced scale invariance as an exact, architecture-independent restriction on the QNN hypothesis class. It also shows that applying a nonlinear preprocessing function is not sufficient by itself to remove the ambiguity. The proposed fixed nonlinear feature map, implemented here as `radial_to_linear`, augments each input with its original values, squared values, and cosine values:

$$
f(x) = \bigl[x,\; x^2,\; \cos(x)\bigr].
$$

This map provably distinguishes all real inputs under subsequent amplitude encoding, adds no trainable parameters, and preserves logarithmic qubit scaling relative to the transformed feature dimension. The experiments compare the proposed encoding against conventional amplitude encoding, angle encoding, and a classical neural-network baseline on three canonical classification datasets and a nonlinear synthetic dataset.

Across the reported tasks, the proposed encoding consistently improves classification accuracy over the quantum encoding baselines by 21–44%. This package is intended to make those comparisons, implementation details, and generated outputs straightforward to reproduce and inspect.

The central experimental comparison is:

- **Conventional amplitude encoding**: compact but invariant to nonzero real input rescaling.
- **Amplitude encoding with `radial_to_linear`**: a fixed nonlinear feature map designed to remove the scale ambiguity before state preparation.
- **Angle encoding**: features are applied as single-qubit rotation angles, providing a comparison with a different resource/representation trade-off.
- **Classical baseline**: a configurable feed-forward neural network trained with PyTorch.

The implementation uses PennyLane for differentiable quantum circuits, PyTorch for model training, and scikit-learn for datasets, preprocessing, metrics, and cross-validation.

## Repository layout

```text
quantum-encoding-public/
├── 2d_visualizations.py   # Generate 2-D decision-boundary figures
├── classical_nn.py        # Classical PyTorch/scikit-learn-compatible baseline
├── main.py                # Main cross-validation benchmark and significance tests
├── qnn.py                 # Quantum encodings, QNN model, and feature transforms
├── requirements.txt       # Python dependencies for the reproducibility environment
├── logs/                  # Runtime logs written by benchmark executions
└── plots/                 # PNG figures written by 2d_visualizations.py
```

### `qnn.py`

This is the reusable quantum-model module. It contains:

- `AngleEncoding`, a custom PennyLane operation that applies one rotation per feature.
- `StackedAngleEncoding`, which reuses qubits for feature vectors longer than the number of qubits and can insert entanglement layers between passes.
- `AmplitudeEncoding`, a thin wrapper around PennyLane state preparation with optional normalization and zero padding.
- `ParameterizedQuantumCircuit`, the trainable ansatz. It supports `x`, `y`, and `z` rotations and three entanglement layouts: `linear`, `reverse_linear`, and `full`.
- `QuantumNeuralNetwork`, a PyTorch module that lazily infers the input size, output size, and required qubit count during `fit()`.
- Feature helpers: `min_max_scale_to_range`, `normalized_arcsin`, `normalized_arccos`, `standardize`, and `radial_to_linear`.

The QNN supports binary and multiclass classification. For binary classification, it returns a single output that is mapped to `[0, 1]` by default; for multiclass classification, it returns one output per class and uses cross-entropy training. The model exposes `fit`, `predict`, `predict_proba`, `draw`, and `draw_mpl` methods.

The module also has a runnable demonstration under its `if __name__ == '__main__':` block. That example trains an amplitude-encoded QNN on a synthetic circles dataset and displays a learned probability surface.

### `classical_nn.py`

Defines `ClassicalNeuralNetwork`, a feed-forward PyTorch classifier that also follows the familiar scikit-learn estimator interface. It can be configured with either:

- a repeated `width` and number of `layers`, or
- explicit `hidden_sizes`, such as `(16, 16)`.

The network uses linear layers, sigmoid hidden activations, and dropout. It supports binary and multiclass targets, minibatch training, optional validation data, Adam optimization, weight decay, and validation-loss early stopping. `main.py` uses the explicit `(16, 16)` configuration as its classical reference model.

### `main.py`

Runs the main benchmark when executed. It evaluates four models on four datasets:

1. Classical neural network with two hidden layers of size 16.
2. QNN with angle encoding and PCA capped at eight features.
3. QNN with raw amplitude encoding.
4. QNN with amplitude encoding preceded by `radial_to_linear`.

The datasets are:

- **Breast Cancer** — scikit-learn's breast-cancer dataset.
- **Wine** — scikit-learn's wine dataset.
- **Digits** — scikit-learn's handwritten-digits dataset.
- **Synthetic NL** — a deterministic 12-feature binary dataset whose labels are generated from nonlinear trigonometric, polynomial, exponential, hyperbolic-tangent, interaction, and noise terms.

For every dataset/model pair, the script:

1. Splits data with six-fold stratified cross-validation.
2. Fits preprocessing only on each training fold.
3. Mean-imputes and standardizes numeric features.
4. One-hot encodes categorical features, if present.
5. Optionally applies PCA for the angle-encoding experiment.
6. Uses Optuna to tune the learning rate, and the classical model's weight decay.
7. Trains the selected model for 40 epochs.
8. Collects out-of-fold predictions and fold-level metrics.

Reported metrics include accuracy, balanced accuracy, macro F1, macro precision, macro recall, and either binary ROC-AUC/AUPR or multiclass one-vs-rest ROC-AUC/macro AUPR when those metrics can be computed.

After model evaluation, `main.py` performs pairwise instance-level significance comparisons using an exact McNemar-style binomial test. The resulting p-values are adjusted with the Holm-Bonferroni procedure.

> **Important:** `main.py` is a script rather than a library module. Its benchmark setup and evaluation loop run at import time, so execute it as a command instead of importing it from another program.

```bash
python3 main.py
```

### `2d_visualizations.py`

Generates the high-resolution three-panel decision-boundary figures presented in the paper for three two-dimensional toy datasets. These figures provide the paper's qualitative visualization of how the different QNN encodings and feature maps shape the learned classification landscapes:

- a roughly linear/blob dataset,
- concentric circles, and
- interleaving moons.

It compares five QNN configurations:

- raw angle encoding,
- angle encoding with standardization,
- raw amplitude encoding,
- amplitude encoding with $(x, e^x)$ feature augmentation, and
- amplitude encoding with `radial_to_linear`, which concatenates the original features, their squares, and their cosines.

Each output figure contains predicted class-1 probability contours, training samples, and training accuracy. PNG files are written to `plots/` with names corresponding to the model configuration.

## How the pieces fit together

The high-level execution flow is:

```text
Dataset
  │
  ▼
Fold-specific preprocessing in main.py
  │
  ├── ClassicalNeuralNetwork
  │
  └── QuantumNeuralNetwork
        │
        ├── optional input transformation
        ├── angle / stacked-angle / amplitude encoding
        ├── optional intermediary entanglement
        └── trainable rotation-and-entanglement ansatz
  │
  ▼
Predictions, metrics, out-of-fold results, and significance tests
```

For the QNN, the number of qubits is inferred as follows unless `num_qubits` is supplied explicitly:

- angle or stacked-angle encoding: enough qubits for the larger of the input and output dimensions;
- amplitude encoding: enough qubits to represent the input vector, while also accommodating the output dimension.

Amplitude encoding requires the transformed input dimension to fit within the state-vector capacity, $2^n$ for $n$ qubits. The model zero-pads inputs when needed and normalizes the encoded state.

## Installation

Create and activate a virtual environment, then install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependencies are listed in [requirements.txt](requirements.txt). The code is written in Python 3.12.6.

## Running the examples

Run the two-dimensional visualization experiment from the repository root:

```bash
python 2d_visualizations.py
```

The script creates `plots/` if necessary and writes five PNG files:

- `qnn_angle_raw.png`
- `qnn_angle_standardize.png`
- `qnn_amplitude_raw.png`
- `qnn_amplitude_exp_4d.png`
- `qnn_amplitude_radial_6d.png`

Run the full benchmark with:

```bash
python main.py
```

A benchmark run creates:

- `results/testing_results_<timestamp>.txt` containing metadata, fold-aggregated metrics, and pairwise significance results;
- `logs/testing_log_<timestamp>.log` containing informational, debug, warning, and exception messages.

The benchmark can be computationally expensive: it combines six outer folds, multiple datasets, four model configurations, and 20 Optuna trials per fold/model. QNN training is performed with PennyLane's `default.qubit` simulator and a PyTorch interface, so runtime depends substantially on the dataset dimensions and available hardware.

### GPU support

The classical neural-network baseline automatically uses CUDA when PyTorch detects a compatible GPU; otherwise it falls back to the CPU. Verify CUDA availability in the active environment with:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

For ease of use and broad compatibility, the QNN experiments use PennyLane's `default.qubit` simulator by default. It runs on the CPU and allows readers to reproduce the experiments without requiring specialized GPU hardware. To run the QNN simulation with PennyLane's CUDA backend instead, install the optional GPU-enabled Lightning plugin in an environment with a compatible NVIDIA GPU and CUDA installation:

```bash
python -m pip install "pennylane-lightning[gpu]"
```

Then change the device construction in `qnn.py` from `default.qubit` to `lightning.gpu`:

```python
device=qml.device('lightning.gpu', wires=self.num_qubits, shots=None)
```

The current repository does not select `lightning.gpu` by default, so the source must be changed locally when GPU execution is desired. The PyTorch `device` argument and the PennyLane simulator device are separate settings; enabling CUDA for the classical baseline does not automatically switch the QNN simulator. `lightning.gpu` requires supported NVIDIA/CUDA hardware.

Run the self-contained QNN demonstration with:

```bash
python qnn.py
```

That command trains the example model in `qnn.py` and displays its Matplotlib decision-boundary visualization.

## Configuration and reproducibility

The benchmark seeds Python's `random` module, NumPy, and PyTorch with `SEED = 69`. The visualization script uses `SEED = 15` for NumPy and PyTorch. Cross-validation uses shuffled, stratified folds with the benchmark seed.

Key benchmark constants in `main.py` include:

| Setting | Value | Purpose |
| --- | ---: | --- |
| `SEED` | `69` | Random seed for benchmark data and training setup |
| `CV_FOLDS` | `6` | Outer stratified cross-validation folds |
| `QNN_OPTUNA_TRIALS` | `20` | Optuna trials per tuned fold/model |
| `QNN_OPTUNA_VAL_SPLIT` | `0.2` | Inner validation fraction for tuning |
| `QNN_ANGLE_PCA_MAX_FEATURES` | `8` | Maximum PCA dimensions for angle encoding |
| `QNN_RADIAL_STANDARDIZE_IN_TRANSFORM` | `False` | Whether `radial_to_linear` standardizes internally |

The QNN benchmark models share a `y`-rotation ansatz, reverse-linear intermediary entanglement, 32 trainable weights, batch size 32, and 40 training epochs. The classical benchmark uses hidden sizes `(16, 16)`, dropout `0.15`, batch size 32, and 40 epochs. Optuna tunes the classical learning rate and weight decay, while the QNN experiments tune learning rate over a higher range.

## Output interpretation

The text result file is organized into:

1. a metadata section describing the run configuration;
2. one section per dataset;
3. one metric summary per model, formatted as mean ± standard deviation over outer folds;
4. a pairwise significance table containing model accuracies, discordant prediction counts, raw exact-test p-values, Holm-adjusted p-values, and a `significant_0.05` flag.

The log file is useful for tracking fold progress and diagnosing failures. In particular, it records the preprocessing/model stage, Optuna progress, fold metrics, and exceptions before the exception is re-raised.

The visualizations should be read as qualitative decision-boundary comparisons.

## Practical notes and limitations

- The benchmark currently assumes classification targets and uses stratified cross-validation.
- The preprocessing pipeline is rebuilt and fitted independently inside every outer fold to avoid using test-fold statistics during training.
- QNN inputs are converted to dense arrays after scikit-learn preprocessing. Very wide one-hot feature matrices can therefore require substantial memory.
- Amplitude encoding has exponential state-vector capacity in the number of qubits. Feature augmentation can increase the input dimension quickly; `radial_to_linear` triples the number of features.
- For ease of use and broad compatibility, the default quantum simulator is PennyLane's `default.qubit`; `lightning.gpu` is an optional CUDA-backed alternative that must be installed and selected manually (see further details in [GPU support](#gpu-support)).
- `classical_layers=True` is available in the QNN implementation, but the current benchmark configurations leave it disabled.
- The current source does not provide automated tests, packaging metadata, or a pinned lockfile. For reproducible research runs, record the Python version and installed package versions alongside the generated result file.
- Generated logs and plots may accumulate across runs. Use separate working copies or archive timestamped outputs when comparing experiments.

## License and citation

The intended license for this repository is the **MIT License**. No `LICENSE` file is included in this package because the repository is kept anonymous for the paper submission and double blind review process.

A formal license file and the associated paper/project citation will be added after the ICLR 2027 review period.
