import math


def normalize_intervals(intervals, mean, std):

    """Normaliza os intervalos usando a mesma transformação do StandardScaler"""

    normalized = []
    for (low, high), m, s in zip(intervals, mean, std):
        low_norm = (low - m) / s
        high_norm = (high - m) / s
        normalized.append((min(low_norm, high_norm), max(low_norm, high_norm)))
    return normalized


# Funções monótonas crescentes -------------------------------------------------------------------


def sigmoid(z):
    if z > 20:  # Evita overflow
        return 1.0
    elif z < -20:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-z))


def elu(z, alpha):
    if z > 0:
        return z
    else:
        return alpha * (math.exp(z) - 1)


def softplus(z):
    if z > 20:
        return z
    return math.log(1.0 + math.exp(z))


# Funções não monótonas --------------------------------------------------------------------


def swish(z):
    return z * sigmoid(z)


# Cálculo do intervalo das funções --------------------------------------------------------------


def sigmoid_interval(z_interval):
    [z_low, z_high] = z_interval
    return (sigmoid(z_low), sigmoid(z_high))


def tanh_interval(z_interval):
    [z_low, z_high] = z_interval
    return (math.tanh(z_low), math.tanh(z_high))


def relu_interval(z_interval):
    [z_low, z_high] = z_interval
    if z_high <= 0:
        return (0.0, 0.0)
    elif z_low >= 0:
        return (z_low, z_high)
    else:
        return (0.0, z_high)


def elu_interval(z_interval, alpha=1.0):
    [z_low, z_high] = z_interval
    return (elu(z_low,alpha), elu(z_high,alpha))


def softplus_interval(z_interval):
    [z_low, z_high] = z_interval
    return (softplus(z_low), softplus(z_high))


def swish_interval(z_interval):
    [z_low, z_high] = z_interval
    values = [swish(z_low), swish(z_high)]
    if z_low <= -1.28 <= z_high:
        values.append(swish(-1.28))
    return (min(values), max(values))