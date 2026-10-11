from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch
from matplotlib.colors import Normalize
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score

from qnn import QuantumNeuralNetwork, radial_to_linear, standardize


SEED = 15
N_SAMPLES = 500
PRECISION = 0.1
SCALING_FACTOR = 5
GRID_BOUNDARY = 2 * SCALING_FACTOR
EPOCHS = 20
LEARNING_RATE = 0.05
BATCH_SIZE = 100
DPI = 300
FONTSIZE_TITLE = 17
FONTSIZE_AXIS = 14
FONTSIZE_TEXT = 14


def custom_map_exp(
    X: torch.Tensor,
) -> torch.Tensor:
    
    # Standardize features
    X = standardize(X)

    # Augment features
    transformed_data = torch.cat([
        X,                    # Original features
        torch.exp(X),         # Exponential features
    ], dim=-1)  # Concatenate along the feature dimension

    return transformed_data


def make_datasets_and_grid():
    X_linear, y_linear = make_classification(
        n_samples=N_SAMPLES,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=15,
        n_clusters_per_class=2,
        scale=0.4,
    )
    X_linear, y_linear = make_blobs(
        n_samples=N_SAMPLES,
        centers=[[-1, -1], [1.5, 0.5]],
        random_state=0,
    )
    rng = np.random.RandomState(2)
    X_linear = X_linear + rng.uniform(size=X_linear.shape)

    moons = make_moons(n_samples=N_SAMPLES, noise=0.3, random_state=0)
    circles = make_circles(n_samples=N_SAMPLES, noise=0.2, factor=0.5, random_state=0)

    x_1, x_2 = np.mgrid[
        -GRID_BOUNDARY : GRID_BOUNDARY + PRECISION : PRECISION,
        -GRID_BOUNDARY : GRID_BOUNDARY + PRECISION : PRECISION,
    ]
    grid = np.column_stack((x_1.ravel(), x_2.ravel()))

    datasets = {
        'linear': (
            X_linear * 0.45 * SCALING_FACTOR - [0.3 * SCALING_FACTOR, 0],
            y_linear,
            grid,
        ),
        'circles': (
            circles[0] * 1.2 * SCALING_FACTOR,
            circles[1],
            grid,
        ),
        'moons': (
            moons[0] * 0.9 * SCALING_FACTOR - [0.43 * SCALING_FACTOR, 0.2 * SCALING_FACTOR],
            moons[1],
            grid,
        ),
    }
    return datasets, x_1, x_2


def build_qnn_configurations():
    common = {
        'output_size': 1,
        'num_weights': 12,
        'rotation_axes': 'y',
        'intermediary_entanglement': True,
        'entanglement': 'reverse_linear',
        'classical_layers': False,
    }

    return [
        {
            'name': 'qnn_angle_raw',
            'title': 'QNN Angle Raw',
            'kwargs': {
                **common,
                'encoding': 'angle',
                'input_transformation': None,
            },
        },
        {
            'name': 'qnn_angle_standardize',
            'title': 'QNN Angle + Standardize',
            'kwargs': {
                **common,
                'encoding': 'angle',
                'input_transformation': standardize,
            },
        },
        {
            'name': 'qnn_amplitude_raw',
            'title': 'QNN Amplitude Raw',
            'kwargs': {
                **common,
                'encoding': 'amplitude',
                'input_transformation': None,
            },
        },
        {
            'name': 'qnn_amplitude_exp_4d',
            'title': 'QNN Amplitude + f(x)=(x, e^x) (4 inputs)',
            'kwargs': {
                **common,
                'encoding': 'amplitude',
                'input_transformation': custom_map_exp,
            },
        },
        {
            'name': 'qnn_amplitude_radial_6d',
            'title': 'QNN Amplitude + RadialToLinear (6 inputs)',
            'kwargs': {
                **common,
                'encoding': 'amplitude',
                'input_transformation': radial_to_linear,
            },
        },
    ]


def run_single_model_on_dataset(model_config, X, y, grid):
    model = QuantumNeuralNetwork(**model_config['kwargs'])
    model.fit(
        X=X,
        y=y,
        output_map=(0, 1),
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        optimizer=torch.optim.Adam,
        debug=False,
    )
    grid_proba = model.predict_proba(grid)[:, 1]
    train_preds = model.predict(X)
    acc = accuracy_score(y, train_preds)
    return grid_proba, acc


def plot_and_save_figure(model_config, datasets, x_1, x_2, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI, constrained_layout=True)
    dataset_order = ['linear', 'circles', 'moons']
    contour_mappable = None

    for axis, dataset_name in zip(axes, dataset_order):
        X, y, grid = datasets[dataset_name]
        grid_proba_flat, acc = run_single_model_on_dataset(model_config, X, y, grid)
        grid_proba = grid_proba_flat.reshape(x_1.shape)

        contour_mappable = axis.contourf(
            x_1,
            x_2,
            np.clip(grid_proba, 0.0, 1.0),
            levels=np.linspace(0.0, 1.0, 26),
            cmap='RdBu',
            vmin=0.0,
            vmax=1.0,
        )
        axis.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            s=55,
            cmap='RdBu',
            vmin=-0.2,
            vmax=1.2,
            edgecolor='white',
            linewidth=1,
        )
        axis.set_title(dataset_name.capitalize(), fontsize=FONTSIZE_TITLE, pad=10)
        axis.set_xlabel('$x_1$', fontsize=FONTSIZE_AXIS)
        axis.set_ylabel('$x_2$', fontsize=FONTSIZE_AXIS)

        acc_text = axis.text(
            0.97,
            0.05,
            f'Accuracy: {acc:.2f}',
            fontsize=FONTSIZE_TEXT,
            ha='right',
            va='bottom',
            transform=axis.transAxes,
            color='black',
        )
        acc_text.set_path_effects([pe.Stroke(linewidth=1.5, foreground='white'), pe.Normal()])

    colorbar_mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap='RdBu')
    colorbar = fig.colorbar(colorbar_mappable, ax=axes, fraction=0.028, pad=0.03)
    colorbar.set_label('Predicted probability (class 1)', fontsize=FONTSIZE_AXIS, labelpad=12)
    colorbar.set_ticks(np.arange(0.0, 1.01, 0.2))
    colorbar.ax.tick_params(labelsize=FONTSIZE_AXIS - 1)

    output_file = output_dir / f"{model_config['name']}.png"
    fig.savefig(output_file, dpi=DPI)
    plt.close(fig)
    return output_file


if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    output_dir = Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets, x_1, x_2 = make_datasets_and_grid()
    model_configs = build_qnn_configurations()

    saved_files = []
    for config in model_configs:
        saved_path = plot_and_save_figure(config, datasets, x_1, x_2, output_dir)
        saved_files.append(saved_path)
        print(f'Saved: {saved_path}')
