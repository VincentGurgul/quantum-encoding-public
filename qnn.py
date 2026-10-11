'''Provides quantum encodings, a trainable QNN, and feature transformations.

The module defines custom PennyLane encoding operations, a configurable
parameterized quantum-circuit ansatz, and ``QuantumNeuralNetwork`` class for binary
and multiclass classification. The QNN infers the required input, output, and
qubit dimensions from training data and supports angle, stacked-angle, and
amplitude encodings. Feature-transformation helpers and a runnable synthetic
decision-boundary demonstration are also included.

Classes:
    AngleEncoding: Apply one rotation angle to each supplied feature.
    StackedAngleEncoding: Reuse qubits for stacked angle features.
    AmplitudeEncoding: Prepare a quantum state from feature amplitudes.
    ParameterizedQuantumCircuit: Build the trainable circuit ansatz.
    QuantumNeuralNetwork: Train and evaluate a PennyLane-based QNN.

Functions:
    min_max_scale_to_range: Scale features to a specified interval.
    normalized_arcsin: Scale features and apply arcsine elementwise.
    normalized_arccos: Scale features and apply arccosine elementwise.
    standardize: Standardize features using their mean and standard deviation.
    radial_to_linear: Augment features with squares and cosine values.
'''

# Environment: .qrl-env
# Python 3.12.6

import re
import warnings
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pennylane as qml
import imageio.v2 as imageio

from pennylane.operation import Operation
from pennylane.ops import StatePrep
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.metrics import roc_auc_score

device = torch.device('cpu')

plt.ioff();


class AngleEncoding(Operation):
    '''Standard Angle Embedding without entanglement layers.

    Args:
        features (tensor_like): Input tensor of shape '(N,)', where N is the
            number of input features to embed, with N <= n.
        wires (Any or Iterable[Any]): Wires that the template acts on.
        rotation (qml.operation.Operator): PennyLane rotation gate class.
        id (str): Custom label given to an operator instance, useful for
            identifying specific instances.
    '''
    def __init__(self, features, wires, rotation, id=None):
        super().__init__(features, wires=wires, id=id)
        self._hyperparameters = {'rotation': rotation}

        n_features = qml.math.shape(features)[-1]
        if n_features > len(wires):
            raise ValueError(
                f'Features must be of length {len(wires)} (number of qubits) or less; got length {n_features}.')

    @staticmethod
    def compute_decomposition(features, wires, rotation):
        '''Representation of the operator as a product of other operators.

        Args:
            features (tensor_like): Input tensor of dimension '(len(wires),)'.
            wires (Any or Iterable[Any]): Wires that the operator acts on.
            rotation (qml.operation.Operator): PennyLane rotation gate class.

        Returns:
            list[.Operator]: Decomposition of the operator.
        '''
        n_features = qml.math.shape(features)[-1]
        batched = qml.math.ndim(features) > 1
        features = qml.math.T(features) if batched else features
        return [rotation(features[i], wires=wires[i]) for i in range(n_features)]
    

class StackedAngleEncoding(Operation):
    '''Stacked Angle Embedding with entanglement layers after encoding all
    qubits.

    Args:
        features (tensor_like): Input tensor of shape '(N,)', where N is the
            number of input features to embed.
        wires (Any or Iterable[Any]): Wires that the template acts on.
        rotation (qml.operation.Operator): PennyLane rotation gate class.
        entanglement (str): Type of entanglement ('full', 'linear', or 'reverse_linear').
    '''
    def __init__(self, features, wires, rotation, entanglement):
        super().__init__(features, wires=wires, id=None)
        self._hyperparameters = {'rotation': rotation, 'entanglement': entanglement}

    @staticmethod
    def compute_decomposition(features, wires, rotation, entanglement):
        '''Representation of the operator as a product of other operators.

        Args:
            features (tensor_like): Input tensor of dimension '(len(wires),)'.
            wires (Any or Iterable[Any]): Wires that the operator acts on.
            rotation (qml.operation.Operator): PennyLane rotation gate class.
            entanglement (str): Type of entanglement ('full', 'linear', or 'reverse_linear').

        Returns:
            list[.Operator]: Decomposition of the operator.
        '''
        operations = []
        num_qubits = len(wires)
        num_features = qml.math.shape(features)[-1]
        batched = qml.math.ndim(features) > 1

        # Iterate through the features in a stacked manner
        for i in range(num_features):
            target_wire = wires[i % num_qubits]
            feature_value = features[..., i] if batched else features[i]
            operations.append(rotation(feature_value, wires=target_wire))

            # Add entanglement layer after each complete pass through all qubits
            if (i + 1) % num_qubits == 0 and entanglement:
                if entanglement == 'linear':
                    for j in range(num_qubits - 1):
                        operations.append(qml.CNOT(wires=[j, j + 1]))
                elif entanglement == 'reverse_linear':
                    for j in range(num_qubits - 1, 0, -1):
                        operations.append(qml.CNOT(wires=[j - 1, j]))
                elif entanglement == 'full':
                    for j in range(num_qubits):
                        if i != j:
                            operations.append(qml.CNOT(wires=[i, j]))

        return operations
    

class AmplitudeEncoding(StatePrep):
    '''Encodes a feature vector into the amplitude vector of an n-qubit quantum
    state.

    Args:
        features (tensor_like): Input tensor of dimension `(2^len(wires),)` or
            less if `pad_with` is specified.
        wires (Any or Iterable[Any]): Wires that the template acts on.
        pad_with (float or complex): Constant to pad the input to size `2^n`,
            if needed.
        normalize (bool): Whether to automatically normalize the input features.
        id (str): Custom label for identifying the operator instance.
        validate_norm (bool): Whether to validate the norm of the input state.
    '''
    def __init__(
            self,
            features,
            wires,
            pad_with=None,
            normalize=False,
            validate_norm=True,
    ):
        super().__init__(
            features,
            wires=wires,
            pad_with=pad_with,
            normalize=normalize,
            validate_norm=validate_norm,
            id=None,
        )


class ParameterizedQuantumCircuit:
    '''Generates a parameterized quantum circuit with specified rotations and
    entanglement patterns that can be used to serve as an ansatz for quantum
    machine learning models. The circuit is defined by rotation gates applied to
    individual qubits and optional entanglement gates between qubits.

    Args:
        num_qubits (int): Disired number of qubits in the quantum circuit.
        rotation_axes (str, optional): Bloch sphere axes to rotate around in the
            parameterized quantum circuit. Can be any combination of 'x', 'y'
            and 'z', e.g. 'xy' or 'xyz'. If multiple letters are passed,
            multiple rotations will happen. Defaults to 'y'.
        entanglement (str, optional): Type of entanglement pattern ('linear',
        'reverse_linear', or 'full'). Defaults to 'reverse_linear'.

    Methods:
        __call__: Constructs the parameterized quantum circuit using the provided weights.
    '''
    def __init__(
        self,
        num_qubits: int,
        rotation_axes: str = 'y',
        entanglement: str = 'reverse_linear',
    ):
        self.num_qubits = num_qubits
        self.rotation_axes = rotation_axes
        self.entanglement = entanglement

    def __call__(self, weights: torch.Tensor):
        '''Applies parameterized rotation gates (RX, RY, RZ) to all qubits
        based on the specified axes, followed by entanglement gates as defined
        by the entanglement pattern.

        Args:
            weights (torch.Tensor): A tensor containing the parameters for the
                rotation gates.
        '''
        for i, param in enumerate(weights.flatten()):

            # Apply rotations to all qubits
            if 'x' in self.rotation_axes:
                qml.RX(param, wires=i % self.num_qubits)
            if 'y' in self.rotation_axes:
                qml.RY(param, wires=i % self.num_qubits)
            if 'z' in self.rotation_axes:
                qml.RZ(param, wires=i % self.num_qubits)

            # After each full pass through all qubits, add entanglement
            if (i + 1) % self.num_qubits == 0 and (i + 1) != len(weights):
                if self.entanglement == 'linear':
                    for j in range(self.num_qubits - 1):
                        qml.CNOT(wires=[j, j + 1])
                elif self.entanglement == 'reverse_linear':
                    for j in range(self.num_qubits - 1, 0, -1):
                        qml.CNOT(wires=[j - 1, j])
                elif self.entanglement == 'full':
                    for i in range(self.num_qubits):
                        for j in range(self.num_qubits):
                            if i != j:
                                qml.CNOT(wires=[i, j])


class QuantumNeuralNetwork(nn.Module):
    '''Quantum Neural Network class using Pennylane and Pytorch.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        num_qubits (int): Number of qubits.
        num_weights (int): Number of weights for the circuit.
        num_qubits (int, optional): Number of qubits to use in the quantum
            circuit. Defaults to None, i.e. minimal number of qubits required
            to encode the provided input size is chosen.
        output_map (tuple, optional): Desired output mapping range. Defaults
            to `None`.
        rotation_axes (str): Bloch sphere axes to rotate around in the
            parameterized quantum circuit. Can be any combination of 'x',
            'y' and 'z', e.g. 'xy' or 'xyz'. If multiple letters are passed,
            multiple rotations will happen. Defaults to 'y'.
        intermediary_entanglement (bool): Whether to add entanglement layers
            between the encoding and the ansatz. Defaults to False.
        classical_layers (bool): Whether to add classical FNN layers before
            and after the variational quantum circuit. Defaults to False.
        output_activation (callable, optional): Activation function to apply 
            to outputs. Defaults to `None`.
        encoding (str): Input encoding strategy ('angle', 'stacked_angle'
            or 'amplitude'). Defaults to 'amplitude'.
        input_transformation (callable, optional): Function to transform input
            with. Defaults to None.
        entanglement (str): Type of entanglement structure to use in between
            rotation layers. Can be 'full', 'linear' or 'reverse_linear'.

    Raises:
        ValueError: If the inputs, outputs, or encoding configurations are
            invalid.
    '''
    def __init__(
        self,
        input_size: int = None,
        output_size: int = None,
        num_weights: int = 12,
        num_qubits: int = None,
        output_map: tuple = None,
        rotation_axes: str = 'y',
        intermediary_entanglement: bool = False,
        classical_layers: bool = False,
        output_activation: callable = None,
        encoding: str = 'amplitude',
        input_transformation: callable = None,
        entanglement: str = 'reverse_linear',
        device: str = 'cpu',
    ):
        '''Initializes a quantum neural network configuration without building
        the underlying PennyLane QNode immediately.

        The actual quantum circuit, ansatz, and Torch layer are created lazily
        during `fit()`, once the input dimensionality, number of target
        classes, and required number of qubits can be inferred from data.

        Args:
            input_size (int, optional): Number of input features expected after
                any input transformation. If omitted, it is inferred during
                fitting.
            output_size (int, optional): Number of output dimensions. For
                classification, this is inferred from `y` during fitting when
                omitted.
            num_weights (int): Number of trainable variational parameters used
                by the parameterized quantum circuit.
            num_qubits (int, optional): Number of qubits to use. If omitted,
                the required number is inferred during fitting based on the
                encoding strategy, transformed input dimension, and number of
                outputs.
            output_map (tuple, optional): Optional output mapping range applied
                to raw expectation values. For binary classification this can
                be set to `(0, 1)`.
            rotation_axes (str): Bloch sphere axes used in the ansatz rotation
                layers. Can contain any combination of `'x'`, `'y'`, and `'z'`.
            intermediary_entanglement (bool): Whether to add an entanglement
                layer between the encoding block and the variational ansatz.
            classical_layers (bool): Whether to return all qubit expectation
                values for use with an additional classical projection layer.
            output_activation (callable, optional): Optional activation
                function applied to the network outputs.
            encoding (str): Encoding strategy to use. Must be one of `'angle'`,
                `'stacked_angle'`, or `'amplitude'`.
            input_transformation (callable, optional): Optional callable that
                transforms the input tensor before quantum encoding.
            entanglement (str): Entanglement topology to use inside the ansatz
                and optional intermediary entanglement layers.
            device (str): Torch device string used for tensors and module
                execution.
        '''
        # Catch erroneous model configurations
        if not bool(re.fullmatch(r'[xyz]*', rotation_axes)):
            raise ValueError(f'Rotation option "{rotation_axes}" not recognized. Please only use the letters "x", "y", or "z".')
        if 'x' not in rotation_axes and 'y' not in rotation_axes:
            print('QNN does not converge with only Z rotations.')
        if entanglement not in ['linear', 'reverse_linear', 'full']:
            raise ValueError(f'Entanglement option {entanglement} not recognized. Please specify "linear", "reverse_linear", or "full".')
        if encoding not in ['angle', 'stacked_angle', 'amplitude']:
            raise ValueError(f'Encoding option {encoding} not recognized. Please specify "angle" or "amplitude".')  

        # Initialize instance and metadata attributes
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_weights = num_weights
        self.num_qubits = num_qubits
        self.output_map = output_map
        self.rotation_axes = rotation_axes
        self.intermediary_entanglement = intermediary_entanglement
        self.classical_layers = classical_layers
        self.output_activation = output_activation
        self.encoding = encoding
        self.input_transformation = input_transformation
        self.entanglement = entanglement
        self.device = device
        self.ansatz = None
        self.quantum_circuit = None
        self.qlayer = None
        self._built = False

    def _infer_num_qubits(self, input_size: int, output_size: int) -> int:
        '''Infers the number of qubits required for the current model setup.

        Args:
            input_size (int): Number of transformed input features presented to
                the quantum encoder.
            output_size (int): Number of output dimensions required by the
                task.

        Returns:
            int: Number of qubits required to represent the inputs and outputs
                under the selected encoding strategy.
        '''
        if self.num_qubits is not None:
            return int(self.num_qubits)
        if self.encoding in ['angle', 'stacked_angle']:
            return int(np.max((input_size, output_size)))
        return int(np.max((int(np.ceil(np.log2(input_size))), output_size)))

    def _build_quantum_model(self):
        '''Builds the PennyLane QNode, variational ansatz, and Torch layer.

        This method assumes that `input_size`, `output_size`, and
        `num_qubits` have already been resolved. It constructs the quantum
        circuit according to the configured encoding strategy and entanglement
        settings, then wraps the QNode as a trainable Torch layer.
        '''
        self.ansatz = ParameterizedQuantumCircuit(
            self.num_qubits, self.rotation_axes, self.entanglement)

        def qnode(inputs, weights):

            if self.encoding == 'angle':
                AngleEncoding(
                    features=inputs,
                    wires=range(self.num_qubits),
                    rotation=qml.RX if 'x' in self.rotation_axes else qml.RY,
                )
            elif self.encoding == 'stacked_angle':
                StackedAngleEncoding(
                    features=inputs,
                    wires=range(self.num_qubits),
                    rotation=qml.RX if 'x' in self.rotation_axes else qml.RY,
                    entanglement=self.entanglement,
                )
            elif self.encoding == 'amplitude':
                AmplitudeEncoding(
                    features=inputs,
                    wires=range(self.num_qubits),
                    normalize=True,
                    pad_with=0,
                )

            if self.intermediary_entanglement:
                if self.entanglement == 'linear':
                    for j in range(self.num_qubits - 1):
                        qml.CNOT(wires=[j, j + 1])
                elif self.entanglement == 'reverse_linear':
                    for j in range(self.num_qubits - 1, 0, -1):
                        qml.CNOT(wires=[j - 1, j])
                elif self.entanglement == 'full':
                    for i in range(self.num_qubits):
                        for j in range(self.num_qubits):
                            if i != j:
                                qml.CNOT(wires=[i, j])

            self.ansatz(weights)

            if self.classical_layers:
                return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
            return [qml.expval(qml.PauliZ(i)) for i in range(self.output_size)]

        self.quantum_circuit = qml.QNode(
            func=qnode,
            device=qml.device('default.qubit', wires=self.num_qubits, shots=None),
            interface='torch',
            diff_method='best',
        )
        self.qlayer = qml.qnn.TorchLayer(
            qnode=self.quantum_circuit,
            weight_shapes={'weights': (self.num_weights,)},
        )
        self._built = True

    def _ensure_built(self):
        '''Verifies that the quantum model has been constructed.

        Raises:
            RuntimeError: If the quantum circuit and Torch layer have not yet
                been created via `fit()`.
        '''
        if not self._built or self.qlayer is None:
            raise RuntimeError('Model is not built yet. Call fit() first to infer shapes and build the quantum circuit.')

    def __map_to_output_range(self, logits):
        '''
        Maps the logits in the interval [-1, 1] to the desired output range.

        Args:
            logits (list, np.ndarray, or tensor): Logits in the interval [-1, 1].

        Returns:
            same type as logits: Values mapped to the set output range.
        '''
        if self.output_map == (-1, 1):
            return logits
        elif self.output_map == (-float('inf'), float('inf')):
            # Apply hyperbolic tangent mapping for infinite range
            return 100 * torch.arctanh(torch.clamp(logits, -0.9999, 0.9999))
        else:
            # Scale logits to fit specified range
            normalized_output = (logits + 1) / 2
            return self.output_map[0] + normalized_output * (self.output_map[1] - self.output_map[0])

    def forward(self, tensor: torch.tensor) -> torch.tensor:
        '''
        Performs a forward pass through the quantum neural network.

        Args:
            input (torch.tensor): Input tensor of shape [batch_size, num_inputs].

        Returns:
            torch.tensor: Output tensor of shape [batch_size, num_outputs].
        '''
        self._ensure_built()
        if self.input_transformation is not None:
            tensor = self.input_transformation(tensor)
        if len(tensor) != self.input_size and tensor.shape[-1] != self.input_size:
            raise ValueError(f'Provided tensor has shape {tensor.shape}, but input size is configured to {self.input_size}')
        if self.classical_layers:
            # tensor = torch.nn.Linear(self.num_inputs, self.num_qubits, dtype=torch.float64)(tensor)
            tensor = self.qlayer(tensor).to(self.device)
            tensor = torch.nn.Linear(self.num_qubits, self.output_size, dtype=torch.float64)(tensor)
        else:
            tensor = self.qlayer(tensor).to(self.device)
        if self.output_map:
            tensor = self.__map_to_output_range(tensor)
        if self.output_activation:
            tensor = self.output_activation(tensor)
        return tensor

    def fit(
        self,
        X,
        y,
        output_map: tuple = None,
        learning_rate: float = 1e-2,
        batch_size: int = 32,
        epochs: int = 100,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        loss_function: callable = None,
        debug: bool = False,
    ):
        '''Fits the quantum neural network to labeled data.

        During fitting, the method converts supported input types to numpy,
        infers the transformed input dimensionality and the number of output
        classes, determines the required number of qubits, builds the quantum
        model lazily, and then trains it using minibatch optimization.

        Args:
            X (pd.DataFrame | pd.Series | torch.Tensor | array_like): Training
                features. If a one-dimensional input is provided, it is treated
                as a single-feature dataset.
            y (pd.DataFrame | pd.Series | torch.Tensor | array_like): Training
                labels. Binary targets are encoded to a single output unit,
                while multiclass targets use one output per class.
            output_map (tuple, optional): Optional output range mapping to use
                during training. For binary classification, `(0, 1)` is the
                typical choice when using `BCELoss`.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Minibatch size used during training.
            epochs (int): Number of training epochs.
            optimizer (torch.optim.Optimizer): Optimizer class used to update
                the model parameters.
            loss_function (callable, optional): Loss function to optimize. If
                omitted, `BCELoss` is used for binary classification and
                `CrossEntropyLoss` for multiclass classification.
            debug (bool): Whether to print average epoch losses during
                training.

        Returns:
            QuantumNeuralNetwork: The fitted model instance.

        Raises:
            ValueError: If amplitude encoding is selected and the transformed
                input dimension exceeds the representational capacity of the
                inferred number of qubits.
        '''
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            X_np = X.to_numpy()
        else:
            X_np = np.asarray(X)

        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        elif isinstance(y, (pd.DataFrame, pd.Series)):
            y_np = y.to_numpy()
        else:
            y_np = np.asarray(y)

        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        if y_np.ndim > 1 and y_np.shape[-1] == 1:
            y_np = y_np.reshape(-1)

        classes, y_encoded = np.unique(y_np, return_inverse=True)
        self._classes = classes
        detected_output_size = 1 if len(classes) == 2 else len(classes)
        X_shape_torch = torch.tensor(X_np, dtype=torch.float64)
        if self.input_transformation is not None:
            transformed_X = self.input_transformation(X_shape_torch)
        else:
            transformed_X = X_shape_torch
        detected_input_size = int(transformed_X.shape[-1])

        self.input_size = detected_input_size
        self.output_size = detected_output_size

        if output_map is not None:
            self.output_map = output_map
        elif detected_output_size == 1 and self.output_map is None:
            self.output_map = (0, 1)

        if detected_output_size > 2:
            self.output_map = None

        inferred_qubits = self._infer_num_qubits(
            input_size=self.input_size,
            output_size=self.output_size,
        )
        if not self.classical_layers:
            inferred_qubits = max(inferred_qubits, self.output_size)
        self.num_qubits = inferred_qubits

        if self.encoding == 'amplitude' and self.input_size > (2 ** self.num_qubits):
            raise ValueError(f'Input dimension {self.input_size} exceeds amplitude capacity for {self.num_qubits} qubits.')

        self._build_quantum_model()

        X_torch = torch.tensor(X_np, dtype=torch.float64).to(self.device)
        is_binary = len(classes) == 2
        if len(classes) == 2:
            y_torch = torch.tensor(y_encoded, dtype=torch.float64).to(self.device).view(-1, 1)
            if loss_function is None:
                loss_function = nn.BCELoss()
        else:
            y_torch = torch.tensor(y_encoded, dtype=torch.long).to(self.device)
            if loss_function is None:
                loss_function = nn.CrossEntropyLoss()

        data_loader = torch.utils.data.DataLoader(
            list(zip(X_torch, y_torch)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        optim = optimizer(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            num_batches = 0
            for features, targets in data_loader:
                optim.zero_grad()
                output = self(features)
                if is_binary and isinstance(loss_function, nn.BCELoss):
                    output = torch.clamp(output, 1e-7, 1 - 1e-7)
                loss = loss_function(output, targets)
                loss.backward()
                optim.step()
                total_loss += loss.item()
                num_batches += 1

            if debug:
                avg_loss = total_loss / max(num_batches, 1)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')

        return self

    def predict_proba(self, y):
        '''Predicts class probabilities for the provided inputs.

        Args:
            y (pd.DataFrame | pd.Series | torch.Tensor | array_like): Input
                samples for which probabilities should be predicted. The
                parameter name is preserved for compatibility with the existing
                interface.

        Returns:
            np.ndarray: Predicted class probabilities. For binary
                classification, the returned array has shape
                `(n_samples, 2)` with columns `[P(class_0), P(class_1)]`. For
                multiclass classification, probabilities are obtained via
                `softmax` across all output dimensions.

        Raises:
            RuntimeError: If called before the model has been built via `fit()`.
        '''
        self._ensure_built()
        if isinstance(y, torch.Tensor):
            X_np = y.detach().cpu().numpy()
        elif isinstance(y, (pd.DataFrame, pd.Series)):
            X_np = y.to_numpy()
        else:
            X_np = np.asarray(y)

        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        X_torch = torch.tensor(X_np, dtype=torch.float64).to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self(X_torch)

        if self.output_size == 1:
            if self.output_map == (0, 1):
                probs_pos = torch.clamp(logits, 0, 1).detach().cpu().numpy().reshape(-1)
            else:
                probs_pos = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            return np.c_[1 - probs_pos, probs_pos]

        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    def predict(self, y):
        '''Predicts class labels for the provided inputs.

        Args:
            y (pd.DataFrame | pd.Series | torch.Tensor | array_like): Input
                samples for which labels should be predicted. The parameter
                name is preserved for compatibility with the existing
                interface.

        Returns:
            np.ndarray: Predicted class labels mapped back to the original
                class values observed during fitting.
        '''
        proba = self.predict_proba(y)

        if self.output_size == 1:
            labels = (proba[:, 1] >= 0.5).astype(int)
            return self._classes[labels] if hasattr(self, '_classes') else labels

        pred_idx = np.argmax(proba, axis=1)
        return self._classes[pred_idx] if hasattr(self, '_classes') else pred_idx

    def draw(self):
        '''Draws the quantum circuit diagram as plain text using PennyLane's drawing functionality.
        '''
        self._ensure_built()
        with warnings.catch_warnings(action='ignore'):
            print(qml.draw(self.quantum_circuit, level='device')(
                np.zeros(self.input_size),
                np.zeros(self.num_weights)))

    def draw_mpl(self):
        '''Draws the quantum circuit diagram in Matplotlib using PennyLane's MPL drawing functionality.
        '''
        self._ensure_built()
        with warnings.catch_warnings(action='ignore'):
            qml.draw_mpl(self.quantum_circuit, level='device')(
                np.zeros(self.input_size),
                np.zeros(self.num_weights))
            plt.show()


def min_max_scale_to_range(
    X: torch.Tensor,
    range: tuple = (-1, 1),
) -> torch.Tensor:
    '''
    Scales the data using min-max normalization to a specified range.

    Args:
        X (torch.Tensor): Input tensor to be scaled.
        range (tuple, optional): Desired range for scaling, specified as
            (min, max). Defaults to (-1, 1).

    Returns:
        torch.Tensor: Scaled tensor with values in the specified range.
            Same dimensions as input.
    '''
    # Normalize to [0, 1]
    X_min, X_max = torch.min(X, dim=0).values, torch.max(X, dim=0).values
    X_normalized = (X - X_min) / (X_max - X_min)
    
    # Scale to [min_val, max_val]
    min_val, max_val = range
    X_scaled = X_normalized * (max_val - min_val) + min_val

    return X_scaled

    
def normalized_arcsin(X: torch.Tensor) -> torch.Tensor:
    '''
    Transforms the data using min-max normalization to [-1, 1] and applies the arcsin function.

    Args:
        X (torch.Tensor): Input tensor to be transformed.

    Returns:
        torch.Tensor: Transformed tensor with values normalized to [-1, 1]
            and then passed through the arcsin function. Same dimensions as input.
    '''
    # Normalize to [-1, 1]
    X = min_max_scale_to_range(X, range=(-1, 1))

    # Apply arcsin
    X = torch.arcsin(X)

    return X
    
def normalized_arccos(X: torch.Tensor) -> torch.Tensor:
    '''
    Transforms the data using min-max normalization to [-1, 1] and applies the arcsin function.

    Args:
        X (torch.Tensor): Input tensor to be transformed.

    Returns:
        torch.Tensor: Transformed tensor with values normalized to [-1, 1]
            and then passed through the arcsin function. Same dimensions as input.
    '''
    # Normalize to [-1, 1]
    X = min_max_scale_to_range(X, range=(-1, 1))

    # Apply arcsin
    X = torch.arccos(X)

    return X


def standardize(
    X: torch.Tensor,
    dof: int = 0,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    '''
    Standardizes the data using z-score normalization.

    Args:
        X (torch.Tensor): Input tensor to be standardized.
        dof (int, optional): Degrees of freedom on which the calculation of
            the standard deviation is to be based on. Value of 1 is unbiased
            estimator, value of 0 is biased estimator. Defaults to 0.
        epsilon (float, optional): Small constant added to the standard
            deviation to avoid division by zero. Defaults to 1e-10.

    Returns:
        torch.Tensor: Standardized tensor with a mean of 0 and a standard
            deviation of 1 for each feature. Same dimensions as input.
    '''
    mean = torch.mean(X, dim=0)
    std = torch.std(X, dim=0, correction=dof)
    return (X - mean) / (std + epsilon)


def radial_to_linear(
    X: torch.Tensor,
    standardize_features: bool = False,
) -> torch.Tensor:
    '''
    Transforms data using direct Cartesian feature augmentations to escape
    radial patterns. Outputs a tensor of original data, original data squared, and cosine. Hence, the output dimension is 3 times the input dimension.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_features).
        standardize_features (bool, optional): Whether to standardize input
            features before augmentation. Defaults to False.

    Returns:
        torch.Tensor: Transformed tensor of shape (n_samples, 3 * n_features).
    '''
    if standardize_features:
        X = standardize(X)

    # Augment features
    transformed_data = torch.cat([
        X,              # Original features
        X**2,           # Quadratic features
        torch.cos(X)    # Cosine features
    ], dim=-1)  # Concatenate along the feature dimension

    return transformed_data


if __name__ == '__main__':

    n_samples=500
    precision = 0.1
    scaling_factor = 5
    grid_boundary = 2 * scaling_factor

    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, random_state=15, n_clusters_per_class=2, scale=0.4,
    )
    X, y = make_blobs(n_samples=n_samples, centers=[[-1, -1], [1.5, 0.5]], random_state=0)
    rng = np.random.RandomState(2)
    X += rng.uniform(size=X.shape)

    moons = make_moons(n_samples=n_samples, noise=0.3, random_state=0)
    circles = make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=0)

    x_1, x_2 = np.mgrid[-grid_boundary:grid_boundary+precision:precision,
                        -grid_boundary:grid_boundary+precision:precision]
    grid = np.column_stack((x_1.ravel(), x_2.ravel()))

    datasets = {
        'moons': (moons[0]*0.9*scaling_factor - [0.43*scaling_factor, 0.2*scaling_factor], moons[1], grid),
        'circles': (circles[0]*1.2*scaling_factor, circles[1], grid),
        'linear': (X*0.45*scaling_factor - [0.3*scaling_factor, 0], y, grid),
    }

    X, y, grid = datasets['circles']

    QNN = QuantumNeuralNetwork(
        input_size=2,
        output_size=1,
        num_weights=12,
        rotation_axes='y',
        intermediary_entanglement=True,
        encoding='amplitude',
        input_transformation=radial_to_linear,
        entanglement='reverse_linear',
    )

    epochs = 20
    plot_epochs = [epochs]

    QNN.fit(
        X=X,
        y=y,
        output_map=(0, 1),
        learning_rate=0.05,
        batch_size=100,
        epochs=epochs,
        optimizer=torch.optim.Adam,
        debug=True,
    )

    grid_proba = QNN.predict_proba(grid)[:, 1].reshape(x_1.shape)
    train_proba = QNN.predict_proba(X)[:, 1]

    grid_preds = [grid_proba]
    auc_values = [roc_auc_score(y, train_proba)]

    _, ax = plt.subplots(figsize=(7, 5), dpi=300)

    ax.set_title(f'QNN with {QNN.num_qubits} Qubits and {QNN.num_weights} Weights, Epochs={plot_epochs[0]}')
    ax.contourf(x_1, x_2, grid_preds[0], 25, cmap='RdBu', vmin=0, vmax=1)
    ax.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu', vmin=-.2, vmax=1.2, edgecolor='white', linewidth=1)
    ax.set(xlabel='$x_1$', ylabel='$x_2$')
    ax.text(0.95, 0.05, f'AUC: {auc_values[0]:.2f}', fontsize=12, ha='right', transform=ax.transAxes)

    plt.show()
