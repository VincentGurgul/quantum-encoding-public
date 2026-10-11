'''Benchmark classical and quantum models on multiple datasets with Optuna tuning.

This script evaluates four models for each dataset using Optuna hyperparameter tuning:
1) Classical NN with fixed hidden layers (16, 16), tuned lr and weight_decay
2) QNN angle encoding, tuned learning_rate
3) QNN amplitude encoding with no transform, tuned learning_rate
4) QNN amplitude encoding with radial-to-linear transform, tuned learning_rate
5) A synthetic high-dimensional nonlinear dataset

Final aggregated metrics are appended iteratively to a timestamped text file.
'''

import random
import logging
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from classical_nn import ClassicalNeuralNetwork
from qnn import QuantumNeuralNetwork, radial_to_linear

SEED = 69
CV_FOLDS = 6
QNN_OPTUNA_TRIALS = 20
QNN_OPTUNA_VAL_SPLIT = 0.2
QNN_RADIAL_STANDARDIZE_IN_TRANSFORM = False
QNN_ANGLE_PCA_MAX_FEATURES = 8

DATASETS = [
    'Breast Cancer',
    'Wine',
    'Digits',
    'Synthetic NL',
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIME_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RESULTS_FILE = RESULTS_DIR / f'testing_results_{TIME_TAG}.txt'

LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / f'testing_log_{TIME_TAG}.log'

logger = logging.getLogger('testing_benchmark')
logger.setLevel(logging.DEBUG)
logger.propagate = False
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

with open(RESULTS_FILE, 'w', encoding='utf-8'):
    pass

with open(RESULTS_FILE, 'a', encoding='utf-8') as file:
    file.write('=' * 80 + '\n')
    file.write('Metadata\n')
    file.write('=' * 80 + '\n')
    file.write(f'Values are averages over {CV_FOLDS}-fold CV ± std.\n')
    file.write(f'Timestamp: {TIME_TAG}\n')
    file.write(f'Seed: {SEED}\n')
    file.write(f'CV folds (outer): {CV_FOLDS}\n')
    file.write(f'Datasets: {"; ".join(DATASETS)}\n')
    file.write('Classical NN params: hidden_sizes=(16,16), dropout=0.15, batch_size=32, epochs=40\n')
    file.write('Classical Optuna search: lr, weight_decay\n')
    file.write('QNN fixed params: rotation_axes=y, intermediary_entanglement=True, entanglement=reverse_linear, num_weights=32, batch_size=32, epochs=40\n')
    file.write('QNN Optuna search: learning_rate only (higher range)\n')
    file.write(f'Optuna: enabled, trials={QNN_OPTUNA_TRIALS}, validation_split={QNN_OPTUNA_VAL_SPLIT}, metric=balanced_accuracy\n')
    file.write('Preprocessing (ALL MODELS): mean imputation + StandardScaler (numeric) + one-hot encoding\n')
    file.write(f'QNN angle PCA: enabled with max_features={QNN_ANGLE_PCA_MAX_FEATURES}\n')
    file.write(
        f'QNN amplitude radial transform: standardize_features={QNN_RADIAL_STANDARDIZE_IN_TRANSFORM}\n\n'
    )

logger.info('Testing benchmark run started')
logger.info('Results file: %s', RESULTS_FILE)
logger.info('Log file: %s', LOG_FILE)
logger.debug('Configuration | seed=%s | cv_folds=%s | datasets=%s', SEED, CV_FOLDS, DATASETS)


def load_dataset(name):
    if name == 'Breast Cancer':
        dataset = load_breast_cancer(as_frame=True)
        return dataset.data, dataset.target
    if name == 'Wine':
        dataset = load_wine(as_frame=True)
        return dataset.data, dataset.target
    if name == 'Digits':
        dataset = load_digits(as_frame=True)
        return dataset.data, dataset.target
    if name == 'Synthetic NL':
        rng = np.random.default_rng(SEED)
        n_samples = 2400
        n_features = 12
        X = rng.uniform(-2.5, 2.5, size=(n_samples, n_features))

        # Nonlinear decision function built from actual nonlinear operations.
        score = (
            1.2 * np.sin(X[:, 0])
            + 0.9 * np.cos(X[:, 1] * X[:, 2])
            - 0.8 * (X[:, 3] ** 2)
            + 0.6 * np.exp(-np.abs(X[:, 4]))
            + 0.5 * np.tanh(X[:, 5] * X[:, 6])
            - 0.4 * np.sin(X[:, 7] + X[:, 8])
            + 0.3 * X[:, 9] * X[:, 10]
            - 0.2 * (X[:, 11] ** 3)
        )
        score += 0.15 * rng.normal(size=n_samples)
        y = (score > np.median(score)).astype(int)

        columns = [f'x{i+1}' for i in range(n_features)]
        return pd.DataFrame(X, columns=columns), pd.Series(y, name='target')
    raise ValueError


def preprocess_all_models(X):
    '''Unified preprocessing for all models: mean imputation + StandardScaler (numeric) + one-hot encoding.'''
    num = X.select_dtypes(include=['number']).columns
    cat = [column for column in X.columns if column not in num]

    numeric = Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('sc', StandardScaler()),
    ])

    categorical = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer([
        ('num', numeric, num),
        ('cat', categorical, cat),
    ])


def to_dense(matrix):
    if hasattr(matrix, 'toarray'):
        return matrix.toarray()
    return np.asarray(matrix)


def reduce_features(X_train, X_test, max_features=4):
    n_features = X_train.shape[1]
    n_components = min(max_features, n_features, X_train.shape[0] - 1)
    if n_components < n_features and n_components >= 1:
        projector = PCA(n_components=n_components, random_state=SEED)
        return projector.fit_transform(X_train), projector.transform(X_test)
    return X_train, X_test


def metric_dict(y, p, prob):
    metrics = {
        'accuracy': accuracy_score(y, p),
        'balanced_accuracy': balanced_accuracy_score(y, p),
        'f1_macro': f1_score(y, p, average='macro'),
        'precision_macro': precision_score(y, p, average='macro', zero_division=0),
        'recall_macro': recall_score(y, p, average='macro', zero_division=0),
    }
    if prob.shape[1] == 2:
        y_arr = np.asarray(y)
        positive_label = np.unique(y_arr)[1]
        y_bin = (y_arr == positive_label).astype(int)
        try:
            metrics['roc_auc'] = roc_auc_score(y_bin, prob[:, 1])
        except Exception as error:
            logger.warning('Failed to compute roc_auc: %s', error)
        try:
            metrics['aupr'] = average_precision_score(y_bin, prob[:, 1])
        except Exception as error:
            logger.warning('Failed to compute aupr: %s', error)
    else:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y, prob, multi_class='ovr')
        except Exception as error:
            logger.warning('Failed to compute roc_auc_ovr: %s', error)
        try:
            y_onehot = pd.get_dummies(np.asarray(y)).values
            metrics['aupr_macro'] = average_precision_score(y_onehot, prob, average='macro')
        except Exception as error:
            logger.warning('Failed to compute aupr_macro: %s', error)
    return metrics


def holm_bonferroni(p_values):
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted_sorted = [0.0] * m
    running_max = 0.0

    for rank, (_, p_value) in enumerate(indexed, start=1):
        adjusted = min((m - rank + 1) * p_value, 1.0)
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank - 1] = running_max

    adjusted = [0.0] * m
    for rank, (original_idx, _) in enumerate(indexed, start=1):
        adjusted[original_idx] = adjusted_sorted[rank - 1]
    return adjusted


def pairwise_instance_significance(y_true, predictions_by_model):
    y_true_arr = np.asarray(y_true)
    model_names = list(predictions_by_model.keys())
    rows = []

    for model_a, model_b in itertools.combinations(model_names, 2):
        pred_a = np.asarray(predictions_by_model[model_a])
        pred_b = np.asarray(predictions_by_model[model_b])
        correct_a = pred_a == y_true_arr
        correct_b = pred_b == y_true_arr

        b_count = int(np.sum(correct_a & ~correct_b))
        c_count = int(np.sum(~correct_a & correct_b))
        discordant = b_count + c_count

        if discordant == 0:
            p_value = 1.0
        else:
            p_value = binomtest(b_count, n=discordant, p=0.5, alternative='two-sided').pvalue

        rows.append({
            'model_a': model_a,
            'model_b': model_b,
            'acc_a': float(np.mean(correct_a)),
            'acc_b': float(np.mean(correct_b)),
            'acc_diff_a_minus_b': float(np.mean(correct_a) - np.mean(correct_b)),
            'b_a_correct_b_wrong': b_count,
            'c_a_wrong_b_correct': c_count,
            'discordant': discordant,
            'p_value': float(p_value),
        })

    if not rows:
        return pd.DataFrame()

    results_df = pd.DataFrame(rows)
    results_df['p_value_holm'] = holm_bonferroni(results_df['p_value'].tolist())
    results_df['significant_0.05'] = results_df['p_value_holm'] < 0.05
    return results_df


def evaluate_model(
    dataset_name,
    model_name,
    X,
    y,
    model_factory,
    use_qnn_preprocessor=False,
    pca_max_features=None,
    tune_with_optuna=False,
    optuna_trials=0,
    optuna_search_space=None,
):
    '''Evaluates one model with stratified cross-validation.

    Args:
        dataset_name (str): Dataset identifier.
        model_name (str): Display name for logging.
        X (pd.DataFrame): Input features.
        y (array_like): Labels.
        model_factory (callable): Callable creating a fresh model instance.
        use_qnn_preprocessor (bool): Whether to use QNN-specific preprocessing.
        pca_max_features (int | None): Optional PCA feature cap. If `None`,
            PCA is skipped.
        tune_with_optuna (bool): Whether to tune this model with Optuna on an
            inner validation split for each outer fold.
        optuna_trials (int): Number of Optuna trials per outer fold.
        optuna_search_space (callable | None): Function mapping an Optuna
            trial to a tuple of `(model_kwargs, fit_overrides)`.

    Returns:
        tuple[pd.Series, np.ndarray, np.ndarray]:
            Mean ± std summary, ground-truth labels, and out-of-fold predictions.
    '''
    outer = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    y_np = np.asarray(y)
    fold_metrics = []
    oof_predictions = np.empty(y_np.shape[0], dtype=object)

    logger.info('[%s] %s started', dataset_name, model_name)

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y_np), start=1):
        logger.info('[%s] %s | Fold %s/%s started', dataset_name, model_name, fold_idx, CV_FOLDS)
        try:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]

            prep = preprocess_all_models(X_train)
            X_train_p = to_dense(prep.fit_transform(X_train))
            X_test_p = to_dense(prep.transform(X_test))

            if pca_max_features is not None:
                X_train_p, X_test_p = reduce_features(
                    X_train_p,
                    X_test_p,
                    max_features=pca_max_features,
                )
                logger.debug(
                    '[%s] %s | Fold %s/%s PCA applied with max_features=%s',
                    dataset_name,
                    model_name,
                    fold_idx,
                    CV_FOLDS,
                    pca_max_features,
                )

            if tune_with_optuna:
                if optuna_search_space is None:
                    raise ValueError('optuna_search_space must be provided when tune_with_optuna=True')

                logger.info('[%s] %s | Fold %s/%s Optuna started (trials=%s)', dataset_name, model_name, fold_idx, CV_FOLDS, optuna_trials)
                X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                    X_train_p,
                    y_train,
                    test_size=QNN_OPTUNA_VAL_SPLIT,
                    random_state=SEED,
                    stratify=y_train,
                )

                def objective(trial):
                    model_kwargs, fit_overrides = optuna_search_space(trial)
                    tuned_model = model_factory(**model_kwargs)
                    tuned_model.fit(X_inner_train, y_inner_train, epochs=40, **fit_overrides)
                    val_pred = tuned_model.predict(X_inner_val)
                    return balanced_accuracy_score(y_inner_val, val_pred)

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
                best_model_kwargs, best_fit_overrides = optuna_search_space(study.best_trial)
                logger.info(
                    '[%s] %s | Fold %s/%s Optuna finished | best_value=%.6f | best_params=%s',
                    dataset_name,
                    model_name,
                    fold_idx,
                    CV_FOLDS,
                    study.best_value,
                    study.best_trial.params,
                )
                model = model_factory(**best_model_kwargs)
            else:
                model = model_factory()

            if tune_with_optuna:
                model.fit(X_train_p, y_train, epochs=40, **best_fit_overrides)
            else:
                model.fit(X_train_p, y_train, epochs=40)
            preds = model.predict(X_test_p)
            probs = model.predict_proba(X_test_p)
            oof_predictions[test_idx] = preds
            metrics = metric_dict(y_test, preds, probs)
            fold_metrics.append(metrics)
            logger.info('[%s] %s | Fold %s/%s finished', dataset_name, model_name, fold_idx, CV_FOLDS)
            logger.debug('[%s] %s | Fold %s/%s metrics=%s', dataset_name, model_name, fold_idx, CV_FOLDS, metrics)
        except Exception as error:
            logger.exception('[%s] %s | Fold %s/%s failed: %s', dataset_name, model_name, fold_idx, CV_FOLDS, error)
            raise

    result_df = pd.DataFrame(fold_metrics)
    summary = result_df.mean().round(4).astype(str) + ' ± ' + result_df.std().round(4).astype(str)

    with open(RESULTS_FILE, 'a', encoding='utf-8') as file:
        file.write(f'---- {model_name} ----\n')
        file.write(summary.to_string())
        file.write('\n\n')

    print(f'[{dataset_name}] {model_name} finished.')
    logger.info('[%s] %s finished', dataset_name, model_name)
    logger.debug('[%s] %s summary=%s', dataset_name, model_name, summary.to_dict())
    return summary, y_np.copy(), oof_predictions


for dataset_name in DATASETS:
    X, y = load_dataset(dataset_name)
    logger.info('Dataset started: %s', dataset_name)

    with open(RESULTS_FILE, 'a', encoding='utf-8') as file:
        file.write('=' * 80 + '\n')
        file.write(f'==== DATASET: {dataset_name} ====\n')
        file.write('=' * 80 + '\n\n')

    print(f'\n=== DATASET: {dataset_name} ===')

    def classical_optuna_space(trial):
        return {
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
        }, {}

    _, y_reference, preds_classical = evaluate_model(
        dataset_name=dataset_name,
        model_name='Classical NN (16, 16)',
        X=X,
        y=y,
        model_factory=lambda **kwargs: ClassicalNeuralNetwork(
            hidden_sizes=(16, 16),
            dropout=0.15,
            batch_size=32,
            epochs=40,
            **kwargs,
        ),
        use_qnn_preprocessor=False,
        tune_with_optuna=True,
        optuna_trials=QNN_OPTUNA_TRIALS,
        optuna_search_space=classical_optuna_space,
    )

    def qnn_optuna_space(trial):
        return {}, {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-1, log=True),
            'batch_size': 32,
        }

    _, _, preds_qnn_angle = evaluate_model(
        dataset_name=dataset_name,
        model_name='QNN Angle (No In-Model Standardize)',
        X=X,
        y=y,
        model_factory=lambda **kwargs: QuantumNeuralNetwork(
            encoding='angle',
            input_transformation=None,
            classical_layers=False,
            rotation_axes='y',
            intermediary_entanglement=True,
            entanglement='reverse_linear',
            num_weights=32,
            **kwargs,
        ),
        use_qnn_preprocessor=True,
        pca_max_features=QNN_ANGLE_PCA_MAX_FEATURES,
        tune_with_optuna=True,
        optuna_trials=QNN_OPTUNA_TRIALS,
        optuna_search_space=qnn_optuna_space,
    )

    _, _, preds_qnn_amp_raw = evaluate_model(
        dataset_name=dataset_name,
        model_name='QNN Amplitude Raw',
        X=X,
        y=y,
        model_factory=lambda **kwargs: QuantumNeuralNetwork(
            encoding='amplitude',
            input_transformation=None,
            classical_layers=False,
            rotation_axes='y',
            intermediary_entanglement=True,
            entanglement='reverse_linear',
            num_weights=32,
            **kwargs,
        ),
        use_qnn_preprocessor=True,
        pca_max_features=None,
        tune_with_optuna=True,
        optuna_trials=QNN_OPTUNA_TRIALS,
        optuna_search_space=qnn_optuna_space,
    )

    _, _, preds_qnn_amp_radial = evaluate_model(
        dataset_name=dataset_name,
        model_name='QNN Amplitude + RadialToLinear',
        X=X,
        y=y,
        model_factory=lambda **kwargs: QuantumNeuralNetwork(
            encoding='amplitude',
            input_transformation=lambda tensor: radial_to_linear(
                tensor,
                standardize_features=QNN_RADIAL_STANDARDIZE_IN_TRANSFORM,
            ),
            classical_layers=False,
            rotation_axes='y',
            intermediary_entanglement=True,
            entanglement='reverse_linear',
            num_weights=32,
            **kwargs,
        ),
        use_qnn_preprocessor=True,
        pca_max_features=None,
        tune_with_optuna=True,
        optuna_trials=QNN_OPTUNA_TRIALS,
        optuna_search_space=qnn_optuna_space,
    )

    predictions_by_model = {
        'Classical NN (16, 16)': preds_classical,
        'QNN Angle (No In-Model Standardize)': preds_qnn_angle,
        'QNN Amplitude Raw': preds_qnn_amp_raw,
        'QNN Amplitude + RadialToLinear': preds_qnn_amp_radial,
    }
    significance_df = pairwise_instance_significance(y_reference, predictions_by_model)

    with open(RESULTS_FILE, 'a', encoding='utf-8') as file:
        file.write('---- Pairwise Significance (instance-level McNemar exact, Holm-corrected) ----\n')
        if significance_df.empty:
            file.write('No pairwise comparisons available.\n\n')
        else:
            file.write(significance_df.round(6).to_string(index=False))
            file.write('\n\n')

    logger.info('[%s] Pairwise significance analysis completed', dataset_name)
    if significance_df.empty:
        logger.warning('[%s] Pairwise significance table is empty', dataset_name)
    else:
        logger.debug('[%s] Pairwise significance results=%s', dataset_name, significance_df.to_dict(orient='records'))

    logger.info('Dataset finished: %s', dataset_name)

logger.info('Testing benchmark run completed successfully')
