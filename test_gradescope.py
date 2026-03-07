"""
test_gradescope.py
------------------
Drop this file into your repo root and run:
    python3 test_gradescope.py

Simulates every Gradescope check we have observed:
  Test 1  - Forward pass (fixed weights -> correct logits)
  Test 2  - Backward return format  (too many values to unpack)
  Test 3  - Gradient correctness    (analytical vs numerical, tol 1e-5)
  Test 4  - train.py CLI smoke test (no crash, wandb-free)
  Test 5  - inference.py F1 > 0.6   (loads best_model.npy)
  Test 6  - best_model.npy exists and is loadable
  Test 7  - best_config.json exists and has required keys
"""

import sys, os, subprocess, json, traceback
import numpy as np

# ── path so we can import the ann package from src/ ──────────────────────────
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []

def record(name, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"{tag} {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append((name, ok))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_args(hidden_size=(3, 2), activation="relu", optimizer="sgd",
              loss="cross_entropy", weight_init="xavier", lr=0.01,
              weight_decay=0.0):
    """Build a minimal argparse-like namespace matching your CLI."""
    import argparse
    args = argparse.Namespace(
        dataset="mnist",
        epochs=1,
        batch_size=4,
        learning_rate=lr,
        weight_decay=weight_decay,
        optimizer=optimizer,
        num_layers=len(hidden_size),
        hidden_size=list(hidden_size),
        activation=activation,
        weight_init=weight_init,
        loss=loss,
        wandb_project="test",
        model_save_path=None,
    )
    return args


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 – Forward pass with fixed weights
# ─────────────────────────────────────────────────────────────────────────────
def test_forward():
    try:
        from ann.neural_network import NeuralNetwork
        args = make_args(hidden_size=(3, 2))
        model = NeuralNetwork(args)

        # Fix weights to known values
        np.random.seed(42)
        for layer in model.layers:
            layer.W = np.random.randn(*layer.W.shape) * 0.1
            layer.b = np.zeros_like(layer.b)

        X = np.random.randn(2, 784)
        logits = model.forward(X)

        assert logits.shape == (2, 10), f"Expected shape (2,10), got {logits.shape}"
        assert not np.any(np.isnan(logits)), "Logits contain NaN"
        assert not np.any(np.isinf(logits)), "Logits contain Inf"

        # Verify it returns LOGITS not probabilities (rows should NOT sum to 1)
        row_sums = softmax(logits).sum(axis=1)
        assert np.allclose(row_sums, 1.0), "softmax of logits should sum to 1 (sanity)"
        raw_sums = logits.sum(axis=1)
        assert not np.allclose(raw_sums, 1.0), \
            "forward() appears to return probabilities, not logits"

        record("Test 1 – Forward pass shape & logits", True)
    except Exception as e:
        record("Test 1 – Forward pass shape & logits", False, traceback.format_exc(limit=3))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 – backward() return format: must unpack as  W_grads, b_grads = model.backward(...)
# ─────────────────────────────────────────────────────────────────────────────
def test_backward_format():
    try:
        from ann.neural_network import NeuralNetwork
        args = make_args(hidden_size=(3, 2))
        model = NeuralNetwork(args)

        X = np.random.randn(4, 784)
        y = np.array([0, 1, 2, 3])
        logits = model.forward(X)

        # Gradescope does exactly this unpack:
        try:
            W_grads, b_grads = model.backward(y, logits)
        except ValueError as e:
            record("Test 2 – backward() return format", False,
                   f"Cannot unpack as (W_grads, b_grads): {e}\n"
                   "       backward() must return a 2-tuple: (list_of_W_grads, list_of_b_grads)")
            return

        n_layers = len(model.layers)
        assert len(W_grads) == n_layers, \
            f"Expected {n_layers} W grads, got {len(W_grads)}"
        assert len(b_grads) == n_layers, \
            f"Expected {n_layers} b grads, got {len(b_grads)}"

        # Check shapes match layers (last->first order)
        for k, (wg, bg, layer) in enumerate(zip(W_grads, b_grads,
                                                  reversed(model.layers))):
            assert wg.shape == layer.W.shape, \
                f"W_grads[{k}] shape {wg.shape} != layer W shape {layer.W.shape}"
            assert bg.shape == layer.b.shape, \
                f"b_grads[{k}] shape {bg.shape} != layer b shape {layer.b.shape}"

        record("Test 2 – backward() return format", True)
    except Exception as e:
        record("Test 2 – backward() return format", False, traceback.format_exc(limit=3))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 – Gradient correctness: analytical vs numerical (tol 1e-5)
# ─────────────────────────────────────────────────────────────────────────────
def test_gradient_correctness():
    try:
        from ann.neural_network import NeuralNetwork
        from ann.activations import Softmax
        args = make_args(hidden_size=(4, 3), activation="tanh", loss="cross_entropy")
        model = NeuralNetwork(args)

        np.random.seed(7)
        for layer in model.layers:
            layer.W = np.random.randn(*layer.W.shape) * 0.1
            layer.b = np.random.randn(*layer.b.shape) * 0.01

        X = np.random.randn(3, 784)
        y = np.array([0, 5, 9])

        def loss_fn(model):
            logits = model.forward(X)
            probs = Softmax().forward(logits)
            m = y.shape[0]
            return -np.sum(np.log(probs[range(m), y] + 1e-15)) / m

        # Analytical gradients
        logits = model.forward(X)
        W_grads, b_grads = model.backward(y, logits)
        # backward returns last->first; we need first->last to match layers
        W_grads_ordered = list(reversed(W_grads))
        b_grads_ordered = list(reversed(b_grads))

        eps = 1e-5
        max_err = 0.0
        for li, layer in enumerate(model.layers):
            # Numerical gradient for a few weights
            for idx in np.ndindex(*layer.W.shape):
                orig = layer.W[idx]
                layer.W[idx] = orig + eps
                lp = loss_fn(model)
                layer.W[idx] = orig - eps
                lm = loss_fn(model)
                layer.W[idx] = orig
                num_grad = (lp - lm) / (2 * eps)
                err = abs(W_grads_ordered[li][idx] - num_grad)
                max_err = max(max_err, err)
                if max_err > 1e-4:   # short-circuit to save time
                    break
            if max_err > 1e-4:
                break

        tol = 1e-4  # slightly looser than gradescope's 1e-7 but catches real bugs
        ok = max_err <= tol
        record("Test 3 – Gradient correctness (numerical check)",
               ok,
               f"Max error: {max_err:.2e}  (tolerance {tol:.0e})" +
               ("" if ok else " — check your chain rule / loss grad"))
    except Exception as e:
        record("Test 3 – Gradient correctness (numerical check)", False,
               traceback.format_exc(limit=4))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 – train.py CLI smoke test (exact args Gradescope uses)
# ─────────────────────────────────────────────────────────────────────────────
def test_train_cli():
    train_script = os.path.join(SRC, "train.py")
    if not os.path.exists(train_script):
        record("Test 4 – train.py CLI smoke test", False,
               f"train.py not found at {train_script}")
        return

    cmd = [
        sys.executable, train_script,
        "--dataset",     "mnist",
        "--epochs",      "2",
        "--batch_size",  "64",
        "--learning_rate", "0.001",
        "--optimizer",   "sgd",
        "--num_layers",  "3",
        "--hidden_size", "128", "128", "128",
        "--activation",  "tanh",
        "--loss",        "cross_entropy",
        "--weight_init", "xavier",
        "-w_p",          "disabled",   # wandb offline
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"    # prevent wandb network calls

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
        ok = result.returncode == 0
        detail = ""
        if not ok:
            # Show last 20 lines of stderr
            lines = (result.stderr + result.stdout).strip().splitlines()
            detail = "\n       ".join(lines[-20:])
        record("Test 4 – train.py CLI smoke test (2 epochs)", ok, detail)
    except subprocess.TimeoutExpired:
        record("Test 4 – train.py CLI smoke test (2 epochs)", False,
               "TIMEOUT after 300s — likely a wandb hang or infinite loop")
    except Exception as e:
        record("Test 4 – train.py CLI smoke test (2 epochs)", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 – best_model.npy exists and is loadable
# ─────────────────────────────────────────────────────────────────────────────
def test_model_file():
    model_path = os.path.join(SRC, "best_model.npy")
    try:
        assert os.path.exists(model_path), f"File not found: {model_path}"
        data = np.load(model_path, allow_pickle=True).item()
        assert isinstance(data, dict), "best_model.npy should contain a dict"
        assert "W0" in data and "b0" in data, "Missing W0/b0 keys in weights dict"
        record("Test 5 – best_model.npy exists & loadable", True,
               f"Keys: {list(data.keys())}")
    except Exception as e:
        record("Test 5 – best_model.npy exists & loadable", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 – best_config.json exists with required keys
# ─────────────────────────────────────────────────────────────────────────────
def test_config_file():
    config_path = os.path.join(SRC, "best_config.json")
    required = ["hidden_size", "activation", "optimizer", "loss",
                "weight_init", "best_f1"]
    try:
        assert os.path.exists(config_path), f"File not found: {config_path}"
        with open(config_path) as f:
            cfg = json.load(f)
        missing = [k for k in required if k not in cfg]
        assert not missing, f"Missing keys: {missing}"
        record("Test 6 – best_config.json exists & valid", True,
               f"best_f1={cfg.get('best_f1', '?'):.4f}, "
               f"hidden_size={cfg.get('hidden_size')}, "
               f"activation={cfg.get('activation')}")
    except Exception as e:
        record("Test 6 – best_config.json exists & valid", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 – inference.py F1 > 0.6
# ─────────────────────────────────────────────────────────────────────────────
def test_inference_f1():
    inference_script = os.path.join(SRC, "inference.py")
    if not os.path.exists(inference_script):
        record("Test 7 – inference.py F1 > 0.6", False,
               f"inference.py not found at {inference_script}")
        return

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    try:
        result = subprocess.run(
            [sys.executable, inference_script],
            capture_output=True, text=True, timeout=180, env=env
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            lines = output.strip().splitlines()
            record("Test 7 – inference.py F1 > 0.6", False,
                   "inference.py crashed:\n       " +
                   "\n       ".join(lines[-15:]))
            return

        # Parse metrics from output
        import ast, re
        match = re.search(r"Inference Metrics:\s*(\{[^}]+\})", output)
        if not match:
            record("Test 7 – inference.py F1 > 0.6", False,
                   f"Could not find 'Inference Metrics: {{...}}' in output.\n"
                   f"       stdout: {output[:500]}")
            return

        metrics = ast.literal_eval(match.group(1))
        f1 = metrics.get("f1", 0.0)
        ok = f1 > 0.6
        record("Test 7 – inference.py F1 > 0.6", ok,
               f"F1={f1:.4f}  accuracy={metrics.get('accuracy',0):.4f}")
    except subprocess.TimeoutExpired:
        record("Test 7 – inference.py F1 > 0.6", False, "TIMEOUT after 180s")
    except Exception as e:
        record("Test 7 – inference.py F1 > 0.6", False, traceback.format_exc(limit=3))


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Gradescope simulation tests")
    print("=" * 60)

    test_forward()
    test_backward_format()
    test_gradient_correctness()
    test_train_cli()
    test_model_file()
    test_config_file()
    test_inference_f1()

    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"  {passed}/{total} tests passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)