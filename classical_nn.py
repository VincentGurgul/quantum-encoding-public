'''Neural network benchmark model components.

This module contains a configurable classical neural network with a
scikit-learn-compatible interface used by the benchmark script.
'''

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class ClassicalNeuralNetwork(BaseEstimator, ClassifierMixin, nn.Module):
    '''Scikit-learn-compatible configurable classical neural network.

    Args:
        width (int): Hidden layer width used when `hidden_sizes` is not
            provided.
        layers (int): Number of hidden layers used when `hidden_sizes` is not
            provided.
        hidden_sizes (tuple[int], optional): Explicit hidden-layer sizes,
            e.g. `(128, 64)`.
        dropout (float): Dropout rate applied after each hidden layer.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Maximum number of training epochs.
        weight_decay (float): Weight decay coefficient.
        patience (int): Early stopping patience measured in epochs.
        device (str, optional): Torch device string. Defaults to CUDA if
            available, otherwise CPU.
    '''
    def __init__(self, width=128, layers=2, hidden_sizes=None, dropout=0.2, lr=1e-3,
                 batch_size=256, epochs=100, weight_decay=1e-4,
                 patience=10, device=None):
        nn.Module.__init__(self)
        self.width = width
        self.layers = layers
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = None
        self.model = None

    def _build_network(self, din, dout):
        '''Builds the internal feedforward network.

        Args:
            din (int): Input feature dimension.
            dout (int): Output dimension.
        '''
        mods = []
        d = din
        layer_sizes = self.hidden_sizes if self.hidden_sizes is not None else tuple([self.width] * self.layers)
        for hidden_size in layer_sizes:
            mods += [nn.Linear(d, hidden_size), nn.Sigmoid(), nn.Dropout(self.dropout)]
            d = hidden_size
        mods.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*mods).to(self.device)
        self.model = self.net

    def forward(self, x):
        '''Runs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Network outputs.
        '''
        return self.net(x)

    def fit(self, X, y, X_val=None, y_val=None, epochs=None):
        '''Fits the neural network classifier.

        Args:
            X (array_like): Training features.
            y (array_like): Training labels.
            X_val (array_like, optional): Validation features.
            y_val (array_like, optional): Validation labels.
            epochs (int, optional): Number of training epochs. If omitted,
                falls back to the value set in `__init__`.

        Returns:
            ClassicalNeuralNetwork: Fitted estimator.
        '''
        X = np.asarray(X, np.float32)
        y = np.asarray(y)
        self.classes_, yenc = np.unique(y, return_inverse=True)
        self.binary = len(self.classes_) == 2
        dout = 1 if self.binary else len(self.classes_)
        self._build_network(X.shape[1], dout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.BCEWithLogitsLoss() if self.binary else nn.CrossEntropyLoss()
        Xt = torch.tensor(X).to(self.device)
        yt = torch.tensor(yenc[:, None] if self.binary else yenc).to(self.device)
        if X_val is not None:
            Xv = torch.tensor(np.asarray(X_val, np.float32)).to(self.device)
            yv = torch.tensor(np.searchsorted(self.classes_, y_val)).to(self.device)
        best = np.inf
        best_state = None
        bad = 0
        num_epochs = self.epochs if epochs is None else epochs
        for _ in range(num_epochs):
            self.model.train()
            idx = torch.randperm(len(Xt))
            for b in idx.split(self.batch_size):
                xb = Xt[b]
                yb = yt[b].float() if self.binary else yt[b]
                opt.zero_grad()
                out = self.model(xb)
                loss = crit(out, yb if self.binary else yb)
                loss.backward()
                opt.step()
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(Xv)
                    if self.binary:
                        val = crit(out, yv[:, None].float()).item()
                    else:
                        val = crit(out, yv).item()
                if val < best:
                    best = val
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        '''Predicts class probabilities.

        Args:
            X (array_like): Input features.

        Returns:
            np.ndarray: Predicted class probabilities.
        '''
        self.model.eval()
        X = torch.tensor(np.asarray(X, np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(X)
            if self.binary:
                p = torch.sigmoid(out).cpu().numpy().ravel()
                return np.c_[1 - p, p]
            return torch.softmax(out, 1).cpu().numpy()

    def predict(self, X):
        '''Predicts class labels.

        Args:
            X (array_like): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        '''
        p = self.predict_proba(X)
        return self.classes_[p.argmax(1)]
