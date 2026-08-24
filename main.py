import numpy as np
from apply_analysis import propagate_through_network
from graphics import sample_random_points, ask_theme


# -----------------------------------------------------------------------
# Intervalos Customizados
# -----------------------------------------------------------------------

custom_intervals = np.array([
    [7.0,   28.0],
    [10.0,  30.0],
    [40.0,  200.0],
    [140.0, 2500.0],
    [0.05,  0.16],
    [0.02,  0.35],
    [0.0,   0.43],
    [0.0,   0.20],
    [0.10,  0.30],
    [0.05,  0.10],
    [0.1,   2.5],
    [0.3,   4.0],
    [0.7,   22.0],
    [6.0,   540.0],
    [0.001, 0.031],
    [0.002, 0.135],
    [0.0,   0.396],
    [0.0,   0.053],
    [0.007, 0.079],
    [0.0,   0.030],
    [7.9,   36.0],
    [12.0,  49.0],
    [50.0,  252.0],
    [185.0, 4254.0],
    [0.07,  0.22],
    [0.027, 1.058],
    [0.0,   1.252],
    [0.0,   0.291],
    [0.15,  0.66],
    [0.055, 0.208],
])


# -----------------------------------------------------------------------
# Saídas brutas
# -----------------------------------------------------------------------

def get_raw_outputs(points):
    from neural_network import pipe
    mlp = pipe.named_steps["mlp"]
    scaler = pipe.named_steps["scaler"]

    X_scaled = scaler.transform(points)
    activation = X_scaled
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        z = activation @ W + b
        if i == len(mlp.coefs_) - 1:
            activation = z
        else:
            act = mlp.activation
            if act == 'tanh':
                activation = np.tanh(z)
            elif act == 'relu':
                activation = np.maximum(0, z)
            elif act == 'logistic':
                activation = 1 / (1 + np.exp(-z))
            else:
                activation = z

    return activation


# -----------------------------------------------------------------------
# Soundness
# -----------------------------------------------------------------------

def verify_soundness(points, output_intervals):
    print("\n[SOUNDNESS] Verificando pontos do gráfico...")
    raw_outputs = get_raw_outputs(points)

    violations = 0
    for point_idx, out_vec in enumerate(raw_outputs):
        for neuron_idx, val in enumerate(out_vec):
            lo, hi = output_intervals[neuron_idx]
            if val < lo - 1e-9 or val > hi + 1e-9:
                violations += 1
                print(f"  [ERRO] Ponto {point_idx}: saída {neuron_idx} = {val:.6f} "
                      f"fora de [{lo:.6f}, {hi:.6f}]")

    n = len(points)
    if violations == 0:
        print(f"  [OK] Todos os {n} pontos têm saídas dentro dos intervalos calculados.")
    else:
        print(f"  [FALHA] {violations} violação(ões) em {n} pontos.")


# -----------------------------------------------------------------------
# Helpers de input
# -----------------------------------------------------------------------

def _ask_neuron(label, n_outputs, default):
    while True:
        raw = input(f"  Neurônio eixo {label} [0-{n_outputs-1}, Enter={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if 0 <= val < n_outputs:
                return val
            print(f"  [ERRO] Digite um valor entre 0 e {n_outputs - 1}.")
        except ValueError:
            print("  [ERRO] Digite um número inteiro.")


def _ask_n_points(default=1000):
    while True:
        raw = input(f"  Número de pontos [Enter={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val > 0:
                return val
            print("  [ERRO] Digite um valor maior que zero.")
        except ValueError:
            print("  [ERRO] Digite um número inteiro.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    from neural_network import X_train
    n_features_dataset = X_train.shape[1]
    n_features_custom = len(custom_intervals)

    print(f"\n[INFO] Dataset atual: {n_features_dataset} features")
    if n_features_custom != n_features_dataset:
        print(f"[AVISO] custom_intervals tem {n_features_custom} entradas, "
              f"mas o dataset tem {n_features_dataset} features.")
    else:
        print(f"[OK] custom_intervals está correto ({n_features_custom} entradas).")

    print("\n1 - Intervalos customizados")
    print("2 - Intervalos do conjunto de treino\n")

    opcao = input("Opção: ").strip()

    if opcao == "1":
        intervals = [(float(row[0]), float(row[1])) for row in custom_intervals]
        output_intervals = propagate_through_network(input_intervals=intervals, verbose=True)
        plot_intervals = intervals

    elif opcao == "2":
        from data import get_data, get_feature_intervals
        X_train = get_data()
        intervals = get_feature_intervals(X_train)
        output_intervals = propagate_through_network(input_intervals=None, verbose=True)
        plot_intervals = intervals

    else:
        print("Opção inválida.")
        return

    n_outputs = len(output_intervals)

    print(f"\n[SAÍDAS] {n_outputs} neurônio(s) de saída:")
    amplitudes = [hi - lo for lo, hi in output_intervals]
    sorted_by_amp = sorted(range(n_outputs), key=lambda i: amplitudes[i], reverse=True)
    for i in range(n_outputs):
        lo, hi = output_intervals[i]
        print(f"  Neurônio {i}: [{lo:.4f}, {hi:.4f}]  (amplitude={amplitudes[i]:.4f})")

    print("\n[CONFIGURAÇÃO DO GRÁFICO]")
    n_points = _ask_n_points(default=1000)

    neuron_x = neuron_y = neuron_z = None
    if n_outputs == 1:
        pass
    elif n_outputs == 2:
        print("\n  Escolha os neurônios para os eixos:")
        neuron_x = _ask_neuron("X", n_outputs, default=sorted_by_amp[0])
        neuron_y = _ask_neuron("Y", n_outputs, default=sorted_by_amp[1])
    else:
        print("\n  Escolha os neurônios para os eixos:")
        neuron_x = _ask_neuron("X", n_outputs, default=sorted_by_amp[0])
        neuron_y = _ask_neuron("Y", n_outputs, default=sorted_by_amp[1])
        neuron_z = _ask_neuron("Z", n_outputs, default=sorted_by_amp[2])

    points = sample_random_points(plot_intervals, n_points=n_points)
    verify_soundness(points, output_intervals)

    # ---- Tema do gráfico ---------------------------------------------
    ask_theme()

    from neural_network import pipe
    activation_name = pipe.named_steps["mlp"].activation.capitalize()

    from graphics import plot_graph
    plot_graph(points, output_intervals,
               neuron_x=neuron_x, neuron_y=neuron_y, neuron_z=neuron_z,
               title=f"Análise Intervalar ({activation_name})")


if __name__ == "__main__":
    main()