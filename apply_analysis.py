from data import get_model, get_feature_intervals, get_scaler, get_data
from functions import addition, multiply
from taylor_series import taylor_polynomial, evaluate_taylor_on_interval
from activation_functions import normalize_intervals, sigmoid_interval, tanh_interval, relu_interval, elu_interval, softplus_interval, swish_interval
import sympy as sp


def apply_activation_interval(act_expr, z_interval, activation_name):
    x = sp.Symbol('x')

    if activation_name == 'logistic':
        return sigmoid_interval(z_interval)
    elif activation_name == 'tanh':
        return tanh_interval(z_interval)
    elif activation_name == 'relu' or act_expr == sp.Max(0, x):
        return relu_interval(z_interval)
    elif activation_name == 'elu':
        return elu_interval(z_interval)
    elif activation_name == 'softplus':
        return softplus_interval(z_interval)
    elif activation_name == 'swish':
        return swish_interval(z_interval)
    elif activation_name == 'identity':
        return z_interval
    else:
        poly_act = taylor_polynomial(act_expr, a=0, order=5)
        return evaluate_taylor_on_interval(poly_act, z_interval)


def propagate_through_network(input_intervals=None, verbose=True):
    """
    Propaga intervalos através da rede neural usando aritmética intervalar.

    Args:
        input_intervals: Lista de tuplas (min, max) para cada feature.
                         Se None, usa os intervalos do dataset de treino.
        verbose: Se True, imprime informações detalhadas da propagação.

    Returns:
        Lista de intervalos de saída — uma tupla (min, max) por neurônio de saída.
    """
    layers = get_model()
    scaler = get_scaler()
    X_train = get_data()

    if input_intervals is None:
        raw_intervals = get_feature_intervals(X_train)
    else:
        raw_intervals = input_intervals

    if verbose:
        print(f"\n[ANÁLISE INTERVALAR]")
        print(f"Features: {len(raw_intervals)}")
        for i, (lo, hi) in enumerate(raw_intervals):
            print(f"  Feature {i:2d}: [{lo:.4f}, {hi:.4f}]")

    current_intervals = normalize_intervals(raw_intervals, scaler.mean_, scaler.scale_)

    for layer_idx, layer in enumerate(layers):
        W = layer["weights"]
        B = layer["bias"]
        act_expr = layer["activation_expr"]
        act_name = layer["activation_name"]

        if verbose:
            print(f"\nCamada {layer_idx + 1}: {W.shape[0]}→{W.shape[1]} ({act_name})")

        next_layer_intervals = []

        for j in range(W.shape[1]):
            z_neuron = (0.0, 0.0)

            for i in range(W.shape[0]):
                w_val = float(W[i, j])
                term = multiply((w_val, w_val), current_intervals[i])
                z_neuron = addition(z_neuron, term)

            bias_val = float(B[j])
            z_neuron = addition(z_neuron, (bias_val, bias_val))

            a_neuron = apply_activation_interval(act_expr, z_neuron, act_name)
            next_layer_intervals.append(a_neuron)

            if verbose and (W.shape[1] <= 6 or j < 3 or j >= W.shape[1] - 3):
                print(f"  N{j:2d}: z=[{z_neuron[0]:.4f}, {z_neuron[1]:.4f}]  a=[{a_neuron[0]:.6f}, {a_neuron[1]:.6f}]")

        current_intervals = next_layer_intervals

    if verbose:
        print("\nSaídas finais:")
        for i, out_interval in enumerate(current_intervals):
            print(f"  Saída {i}: [{out_interval[0]:.6f}, {out_interval[1]:.6f}]")

    return current_intervals