"""Tests for gingado.dnvs module."""

import os
import tempfile

import numpy as np
import pytest

os.environ["KERAS_BACKEND"] = "torch"

import keras
import keras.layers as L

from gingado.dnvs import (
    DNVS,
    DNVS_Q,
    TemperatureScaledSoftmax,
    prepare_Xy,
    quantile_loss,
)


@pytest.fixture
def synthetic_3d_data():
    """Create small synthetic 3D data for DNVS testing."""
    np.random.seed(42)
    n_samples, n_timesteps, n_features = 20, 10, 4
    X = np.random.randn(n_samples, n_timesteps, n_features).astype(np.float32)
    y = np.random.randn(n_samples, n_timesteps, 1).astype(np.float32)
    return X, y


@pytest.fixture
def fit_args_fast():
    """Minimal fit args for fast tests."""
    return {
        "epochs": 3,
        "batch_size": 10,
        "shuffle": False,
        "verbose": 0,
    }


class TestTemperatureScaledSoftmax:
    def test_temperature_is_trainable(self):
        """Temperature should change during training."""
        inputs = keras.Input(shape=(5,))
        x = L.Dense(10, activation="relu")(inputs)
        x = TemperatureScaledSoftmax(start_value=1.0)(x)
        model = keras.Model(inputs, x)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        np.random.seed(42)
        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = keras.utils.to_categorical(
            np.random.randint(0, 10, 100), num_classes=10
        )

        initial_temp = float(model.layers[-1].temperature.numpy())
        model.fit(X_train, y_train, epochs=30, verbose=0)
        trained_temp = float(model.layers[-1].temperature.numpy())

        assert not np.isclose(trained_temp, initial_temp), (
            "Temperature did not change during training"
        )

    def test_serialization(self):
        """Model with TemperatureScaledSoftmax should save and reload correctly."""
        inputs = keras.Input(shape=(5,))
        x = L.Dense(10, activation="relu")(inputs)
        x = TemperatureScaledSoftmax(start_value=2.0)(x)
        model = keras.Model(inputs, x)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        np.random.seed(42)
        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = keras.utils.to_categorical(
            np.random.randint(0, 10, 50), num_classes=10
        )
        model.fit(X_train, y_train, epochs=5, verbose=0)

        trained_temp = float(model.layers[-1].temperature.numpy())

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_temp = float(loaded_model.layers[-1].temperature.numpy())
        assert np.isclose(trained_temp, loaded_temp), (
            "Loaded temperature does not match trained value"
        )

    def test_custom_start_value(self):
        """Layer should initialize with the given start_value."""
        layer = TemperatureScaledSoftmax(start_value=2.0)
        layer.build((None, 5))
        assert float(layer.temperature.numpy()) == pytest.approx(2.0)


class TestPrepareXy:
    def test_output_shapes(self):
        """Output shapes should follow (N - context, context, F) convention."""
        N, F = 100, 5
        context = 36
        X = np.random.randn(N, F).astype(np.float32)
        y = np.random.randn(N, 1).astype(np.float32)

        X_out, y_out = prepare_Xy(X, y, context=context)

        assert X_out.shape == (N - context, context, F)
        assert y_out.shape == (N - context, context, 1)

    def test_1d_y_input(self):
        """Should handle 1D y input."""
        N, F = 50, 3
        context = 10
        X = np.random.randn(N, F).astype(np.float32)
        y = np.random.randn(N).astype(np.float32)

        X_out, y_out = prepare_Xy(X, y, context=context)

        assert X_out.shape == (N - context, context, F)
        assert y_out.shape == (N - context, context, 1)


class TestDNVS:
    def test_fit_predict(self, synthetic_3d_data, fit_args_fast):
        """DNVS should fit and predict with correct output shape."""
        X, y = synthetic_3d_data
        model = DNVS(latent_dim=8, varsel_dim=4, dropout=0.0, gauss_noise=0.0)
        model.fit(X, y, fit_args=fit_args_fast)
        preds = model.predict(X)

        assert preds.shape == y.shape

    def test_get_variable_weights_shape(self, synthetic_3d_data, fit_args_fast):
        """Variable weights should have same shape as X and sum to 1 along features."""
        X, y = synthetic_3d_data
        model = DNVS(latent_dim=8, varsel_dim=4, dropout=0.0, gauss_noise=0.0)
        model.fit(X, y, fit_args=fit_args_fast)

        weights = model.get_variable_weights(X)

        assert weights.shape == X.shape
        # Weights should sum to ~1 along features axis
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_get_learned_temperature(self, synthetic_3d_data, fit_args_fast):
        """Should return a scalar temperature value."""
        X, y = synthetic_3d_data
        model = DNVS(latent_dim=8, varsel_dim=4, dropout=0.0, gauss_noise=0.0)
        model.fit(X, y, fit_args=fit_args_fast)

        temp = model.get_learned_temperature()
        assert temp.shape == ()

    def test_validation_data(self, synthetic_3d_data, fit_args_fast):
        """Should accept validation_data without error."""
        X, y = synthetic_3d_data
        model = DNVS(latent_dim=8, varsel_dim=4, dropout=0.0, gauss_noise=0.0)
        # Use same data as validation for simplicity
        model.fit(X, y, validation_data=(X, y), fit_args=fit_args_fast)
        assert hasattr(model, "model_")


class TestDNVSQ:
    def test_multi_quantile_prediction(self, synthetic_3d_data, fit_args_fast):
        """DNVS_Q should return one prediction per quantile."""
        X, y = synthetic_3d_data
        quantiles = [0.1, 0.5, 0.9]
        model = DNVS_Q(
            quantiles=quantiles,
            latent_dim=8,
            varsel_dim=4,
            dropout=0.0,
            gauss_noise=0.0,
        )
        model.fit(X, y, fit_args=fit_args_fast)
        preds = model.predict(X)

        assert len(preds) == len(quantiles)
        for p in preds:
            assert p.shape == y.shape

    def test_quantile_specific_weights(self, synthetic_3d_data, fit_args_fast):
        """Should return weights for a specific quantile branch."""
        X, y = synthetic_3d_data
        model = DNVS_Q(
            quantiles=[0.1, 0.5, 0.9],
            latent_dim=8,
            varsel_dim=4,
            dropout=0.0,
            gauss_noise=0.0,
        )
        model.fit(X, y, fit_args=fit_args_fast)

        weights = model.get_variable_weights(X, quantile=0.5)
        assert weights.shape == X.shape
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)


# --- quantile_loss tests ---


class TestQuantileLoss:
    def test_asymmetry(self):
        """High quantile should penalize under-predictions more."""
        y_true = keras.ops.array([1.0, 2.0, 3.0, 4.0])

        # Under-predictions (y_pred < y_true)
        y_pred_under = keras.ops.array([0.0, 1.0, 2.0, 3.0])
        # Over-predictions (y_pred > y_true)
        y_pred_over = keras.ops.array([2.0, 3.0, 4.0, 5.0])

        loss_fn = quantile_loss(0.9)
        loss_under = float(loss_fn(y_true, y_pred_under))
        loss_over = float(loss_fn(y_true, y_pred_over))

        assert loss_under > loss_over, "q=0.9 should penalize under-predictions more"

    def test_median_equals_half_mae(self):
        """quantile_loss(0.5) should equal 0.5 * MAE."""
        y_true = keras.ops.array([1.0, 2.0, 3.0, 4.0])
        y_pred = keras.ops.array([1.5, 2.5, 2.0, 5.0])

        loss_fn = quantile_loss(0.5)
        loss_val = float(loss_fn(y_true, y_pred))

        mae = float(keras.ops.mean(keras.ops.abs(y_true - y_pred)))
        assert np.isclose(loss_val, 0.5 * mae, atol=1e-6)


# --- scikit-learn API compliance tests ---


class TestSklearnAPI:
    def test_get_params(self):
        """get_params should return all constructor arguments."""
        model = DNVS(latent_dim=64, varsel_dim=16)
        params = model.get_params()

        assert params["latent_dim"] == 64
        assert params["varsel_dim"] == 16
        assert "dropout" in params
        assert "start_temp" in params
        assert "gauss_noise" in params
        assert "learning_rate" in params

    def test_set_params(self):
        """set_params should update attributes."""
        model = DNVS()
        model.set_params(latent_dim=64)
        assert model.latent_dim == 64

    def test_get_params_dnvs_q(self):
        """DNVS_Q get_params should include quantiles."""
        model = DNVS_Q(quantiles=[0.25, 0.75])
        params = model.get_params()
        assert params["quantiles"] == [0.25, 0.75]
