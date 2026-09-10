import numpy as np
import pandas as pd

def _ok(task):
    print(f"{task}: All tests passed!")


def test_A0(func, df):
    outputs = func(df)
    assert len(outputs) == 4, "Return X_train, X_test, y_train, y_test."
    X_train, X_test, y_train, y_test = outputs

    assert isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray), "Feature sets must be NumPy arrays."
    assert X_train.ndim == 2 and X_test.ndim == 2, "Feature arrays must be 2D."
    assert X_train.shape[1] == df.shape[1] - 1, "X should contain all columns except the target."
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test), "Feature/label lengths do not match."
    assert len(X_train) + len(X_test) == len(df), "Train and test sizes must sum to the full dataset length."
    assert abs(len(X_test) / len(df) - 0.2) < 1e-9, "Use test_size=0.2."

    # Reproducibility / random_state=42
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["price"]).to_numpy(dtype=float)
    y = df["price"].to_numpy(dtype=float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    assert np.allclose(X_train, Xtr) and np.allclose(X_test, Xte), "Split does not match test_size=0.2, random_state=42."
    assert np.allclose(y_train, ytr) and np.allclose(y_test, yte), "Target split is incorrect."
    _ok("A0")

def test_A1(func):
    X = np.array([[1., 2.], [3., 4.], [-1., 5.]])
    w = np.array([0.5, -2.0])
    b = 1.25
    out = np.asarray(func(X, w, b))
    expected = X @ w + b
    assert out.shape == (3,), "Output must have shape (m,)."
    assert np.allclose(out, expected), "Prediction values are incorrect."
    _ok("A1")

def test_A2(func):
    y_true = np.array([2., -1., 4., 7.])
    y_pred = np.array([1., 1., 5., 3.])
    out = func(y_true, y_pred)
    expected = np.mean((y_true - y_pred) ** 2)
    assert np.isscalar(out), "MSE must be a scalar."
    assert np.isclose(out, expected), "MSE value is incorrect."
    _ok("A2")

def test_A3(func):
    X = np.array([[1., 2.], [2., -1.], [0.5, 3.]])
    y = np.array([4., -1., 5.])
    w = np.array([0.3, -0.2])
    b = 0.7
    dw, db = func(X, y, w, b)

    pred = X @ w + b
    err = pred - y
    m = len(y)
    expected_dw = (2 / m) * (X.T @ err)
    expected_db = (2 / m) * np.sum(err)

    assert np.asarray(dw).shape == w.shape, "dw has the wrong shape."
    assert np.isscalar(db), "db must be a scalar."
    assert np.allclose(dw, expected_dw), "dw is incorrect."
    assert np.isclose(db, expected_db), "db is incorrect."
    _ok("A3")

def test_A4(func):
    w = np.array([1.0, -2.0])
    b = 0.5
    dw = np.array([0.2, -0.4])
    db = -0.3
    lr = 0.1
    new_w, new_b = func(w.copy(), b, dw, db, lr)
    assert np.allclose(new_w, np.array([0.98, -1.96])), "w update is incorrect."
    assert np.isclose(new_b, 0.53), "b update is incorrect."
    _ok("A4")

def test_A5(func):
    X = np.array([[-2.], [-1.], [0.], [1.], [2.]])
    y = 3.0 * X[:, 0] + 2.0
    w, b, history = func(X, y, learning_rate=0.05, num_iterations=300)
    history = np.asarray(history)

    assert np.asarray(w).shape == (1,), "w has the wrong shape."
    assert np.isscalar(b), "b must be a scalar."
    assert history.shape == (300,), "cost_history must contain one value per iteration."
    assert np.all(np.isfinite(history)), "cost_history contains non-finite values."
    assert history[-1] < history[0], "Training cost did not decrease."
    assert np.allclose(w[0], 3.0, atol=0.1), "Learned slope is not close enough."
    assert np.isclose(b, 2.0, atol=0.1), "Learned bias is not close enough."
    _ok("A5")

def test_A6(func):
    X = np.array([[1.], [2.], [3.], [4.]], dtype=float)
    y = np.array([3., 5., 7., 9.])
    lrs = [0.001, 0.01, 0.05]
    histories = func(X, y, lrs, num_iterations=30)

    assert isinstance(histories, dict), "Return value must be a dictionary."
    assert set(histories.keys()) == set(lrs), "Dictionary keys must match the learning rates."
    for lr in lrs:
        h = np.asarray(histories[lr])
        assert h.shape == (30,), "Each history must contain num_iterations values."
        assert np.all(np.isfinite(h)), "Cost history contains non-finite values."
    _ok("A6")

def test_A7(func):
    X = np.array([
        [10., 1., 100.],
        [20., 2., 300.],
        [30., 3., 500.],
        [40., 4., 700.]
    ])
    X_std, X_mm, std_scaler, mm_scaler = func(X)

    assert X_std.shape == X.shape and X_mm.shape == X.shape, "Scaled arrays must keep the original shape."
    assert np.allclose(np.mean(X_std, axis=0), 0.0, atol=1e-10), "Standard-scaled features should have mean 0."
    assert np.allclose(np.std(X_std, axis=0), 1.0, atol=1e-10), "Standard-scaled features should have std 1."
    assert np.allclose(np.min(X_mm, axis=0), 0.0), "MinMax-scaled feature minimums should be 0."
    assert np.allclose(np.max(X_mm, axis=0), 1.0), "MinMax-scaled feature maximums should be 1."
    assert hasattr(std_scaler, "transform") and hasattr(mm_scaler, "transform"), "Return fitted scaler objects."
    _ok("A7")

def test_A8(func):
    X = np.array([[0.], [1.], [2.], [3.], [4.]])
    y = 4 * X[:, 0] - 3
    model, mse = func(X, y)

    assert hasattr(model, "predict"), "Return a fitted model as the first output."
    assert np.isscalar(mse), "MSE must be a scalar."
    assert mse < 1e-12, "LinearRegression should fit this exact linear dataset almost perfectly."
    _ok("A8")

def test_A9(func):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold, cross_val_score

    rng = np.random.default_rng(7)
    X = np.linspace(-3, 3, 120).reshape(-1, 1)
    y = 1.5 + 2.0 * X[:, 0] - 0.9 * X[:, 0]**2 + rng.normal(0, 0.15, len(X))
    degrees = np.arange(1, 6)

    best_degree, mean_scores = func(X, y, degrees, n_splits=5)
    mean_scores = np.asarray(mean_scores)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    expected_scores = []
    for degree in degrees:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg", LinearRegression())
        ])
        expected_scores.append(
            cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
        )
    expected_scores = np.asarray(expected_scores)
    expected_best = degrees[np.argmax(expected_scores)]

    assert mean_scores.shape == (len(degrees),), "Return one mean score for each degree."
    assert np.all(np.isfinite(mean_scores)), "CV scores contain non-finite values."
    assert np.allclose(mean_scores, expected_scores), "The returned CV mean scores are incorrect."
    assert best_degree == expected_best, "best_degree does not match the largest mean CV score."
    _ok("A9")

def test_B0(func, df):
    outputs = func(df)
    assert len(outputs) == 5, "Return X_train_scaled, X_test_scaled, y_train, y_test, scaler."
    Xtr, Xte, ytr, yte, scaler = outputs

    assert isinstance(Xtr, np.ndarray) and isinstance(Xte, np.ndarray), "Scaled feature sets must be NumPy arrays."
    assert Xtr.ndim == 2 and Xte.ndim == 2, "Feature arrays must be 2D."
    assert len(Xtr) == len(ytr) and len(Xte) == len(yte), "Feature/label lengths do not match."
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-10), "Scaler does not appear to be fitted on the training set."
    assert hasattr(scaler, "mean_"), "Return the fitted StandardScaler."
    _ok("B0")

def test_B1(func):
    d = func(np.array([1., 2.]), np.array([4., 6.]))
    assert np.isscalar(d), "Distance must be a scalar."
    assert np.isclose(d, 5.0), "Euclidean distance is incorrect."
    d2 = func(np.array([-1., 0., 2.]), np.array([-1., 0., 2.]))
    assert np.isclose(d2, 0.0), "Distance from a point to itself must be 0."
    _ok("B1")

def test_B2(func):
    X = np.array([[0., 0.], [3., 4.], [6., 8.]])
    q = np.array([0., 0.])
    out = np.asarray(func(X, q))
    assert out.shape == (3,), "Return one distance per training sample."
    assert np.allclose(out, np.array([0., 5., 10.])), "Distance array is incorrect."
    _ok("B2")

def test_B3(func):
    X = np.array([[0., 0.], [5., 0.], [1., 0.], [3., 0.]])
    q = np.array([0., 0.])
    idx = np.asarray(func(X, q, 3))
    assert idx.shape == (3,), "Return exactly k indices."
    assert np.array_equal(idx, np.array([0, 2, 3])), "Nearest-neighbor indices are incorrect."
    _ok("B3")

def test_B4(func):
    X = np.array([[0.], [5.], [1.], [3.]])
    y = np.array([1, 0, 1, 0])
    labels = np.asarray(func(X, y, np.array([0.]), 3))
    assert np.array_equal(labels, np.array([1, 1, 0])), "Neighbor labels are incorrect."
    _ok("B4")

def test_B5(func):
    p0, p1 = func(np.array([1, 0, 1, 1, 0]))
    assert np.isclose(p0, 0.4) and np.isclose(p1, 0.6), "Class probabilities are incorrect."
    assert np.isclose(p0 + p1, 1.0), "Probabilities must sum to 1."
    _ok("B5")

def test_B6(func):
    X = np.array([[0.], [1.], [2.], [8.], [9.]])
    y = np.array([0, 0, 0, 1, 1])
    pred, p1 = func(X, y, np.array([1.5]), 3)
    assert pred in (0, 1), "Prediction must be 0 or 1."
    assert np.isclose(p1, 0.0), "Class-1 probability is incorrect."
    assert pred == 0, "Predicted class is incorrect."
    _ok("B6")

def test_B7(func):
    Xtr = np.array([[0.], [1.], [2.], [8.], [9.]])
    ytr = np.array([0, 0, 0, 1, 1])
    Xte = np.array([[0.5], [8.5]])
    pred, prob = func(Xtr, ytr, Xte, 3)
    pred, prob = np.asarray(pred), np.asarray(prob)
    assert pred.shape == (2,) and prob.shape == (2,), "Return one prediction and probability per test sample."
    assert np.array_equal(pred, np.array([0, 1])), "Batch predictions are incorrect."
    assert np.all((prob >= 0) & (prob <= 1)), "Probabilities must be between 0 and 1."
    _ok("B7")

def test_B8(func):
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0])
    cm = np.asarray(func(y_true, y_pred))
    expected = np.array([[2, 1], [2, 2]])
    assert cm.shape == (2, 2), "Confusion matrix must have shape (2, 2)."
    assert np.array_equal(cm, expected), "Confusion matrix values are incorrect."
    _ok("B8")

def test_B9(func):
    probs = np.array([0.1, 0.49, 0.5, 0.81])
    out = np.asarray(func(probs, 0.5))
    assert np.array_equal(out, np.array([0, 0, 1, 1])), "Thresholding is incorrect."
    assert np.issubdtype(out.dtype, np.integer), "Predictions should be integer labels."
    _ok("B9")

def test_B10(func):
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold, cross_val_score

    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=12
    )
    k_values = np.arange(1, 8)
    best_k, mean_scores = func(X, y, k_values, n_splits=4)
    mean_scores = np.asarray(mean_scores)

    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    expected_scores = []
    for k in k_values:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])
        expected_scores.append(
            cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        )
    expected_scores = np.asarray(expected_scores)
    expected_best = k_values[np.argmax(expected_scores)]

    assert mean_scores.shape == (len(k_values),), "Return one mean score per k."
    assert np.all((mean_scores >= 0) & (mean_scores <= 1)), "Accuracy scores must be between 0 and 1."
    assert np.allclose(mean_scores, expected_scores), "The returned CV mean scores are incorrect."
    assert best_k == expected_best, "best_k does not match the largest mean CV score."
    _ok("B10")
