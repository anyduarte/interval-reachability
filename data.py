from neural_network import *
import sympy as sp


def get_model():
    mlp = pipe.named_steps["mlp"]

    x = sp.Symbol('x')
    activation_map = {
        'logistic': 1 / (1 + sp.exp(-x)),
        'tanh': sp.tanh(x),
        'relu': sp.Max(0, x),
        'identity': x,
        'swish': x / (1 + sp.exp(-x)),
        'elu': sp.Piecewise((sp.exp(x) - 1, x < 0), (x, True)),
        'softplus': sp.log(1 + sp.exp(x)),
        'sigmoid_scaled': 2 / (1 + sp.exp(-x)) - 1,
    }

    activation_expr = activation_map.get(mlp.activation, x)

    layers = []
    num_layers = len(mlp.coefs_)

    for i in range(num_layers):
        if hasattr(mlp, 'out_activation_') and i == num_layers - 1:
            # MLPRegressor: última camada é sempre identity
            current_activation_expr = x
            current_activation_name = 'identity'
        else:
            # MLPClassifier ou camadas ocultas do MLPRegressor
            current_activation_expr = activation_expr
            current_activation_name = mlp.activation

        layers.append({
            "weights": mlp.coefs_[i],
            "bias": mlp.intercepts_[i],
            "activation_expr": current_activation_expr,
            "activation_name": current_activation_name
        })

    return layers


def get_feature_intervals(X):
    return list(zip(X.min(axis=0), X.max(axis=0)))


def get_data():
    return X_train


def get_scaler():
    return pipe.named_steps["scaler"]