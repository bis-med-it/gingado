"""Deep Neural Variable Selection (DNVS) estimators.

Implements the DNVS architecture from Aquilina et al. (BIS Working Paper 1291, 2025)
for simultaneous prediction and time-varying variable importance estimation.
"""

import keras
import keras.layers as L
import numpy as np
from sklearn.base import BaseEstimator

__all__ = [
    "TemperatureScaledSoftmax",
    "DNVS",
    "DNVS_Q",
    "quantile_loss",
    "prepare_Xy",
]


def prepare_Xy(
    X: np.ndarray, y: np.ndarray, context: int = 36
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 2D arrays into 3D sliding-window sequences for RNN input.

    Args:
        X: Input features of shape (N, F).
        y: Target of shape (N,) or (N, 1).
        context: Number of time steps per window.

    Returns:
        Tuple of (X_out, y_out) with shapes (N - context, context, F) and
        (N - context, context, 1).

    Raises:
        ValueError: If context is not greater than 0 and less than the number
            of samples.
    """
    if context <= 0 or context >= X.shape[0]:
        raise ValueError(
            "context must be greater than 0 and less than the number of samples"
        )
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    num_samples = X.shape[0] - context
    samples_X = []
    samples_y = []
    for i in range(num_samples):
        samples_X.append(X[i : i + context])
        samples_y.append(y[i : i + context])
    return np.stack(samples_X, axis=0), np.stack(samples_y, axis=0)


def quantile_loss(q: float):
    """Return a quantile (pinball) loss function for a given quantile q.

    Args:
        q: Quantile value in (0, 1).

    Returns:
        A Keras-compatible loss function.
    """

    def loss(y_true, y_pred):
        error = keras.ops.subtract(y_true, y_pred)
        return keras.ops.mean(keras.ops.maximum(q * error, (q - 1) * error))

    loss.__name__ = f"quantile_loss_{int(q * 100)}"
    return loss


@keras.saving.register_keras_serializable()
class TemperatureScaledSoftmax(keras.Layer):
    """Softmax layer with a learnable temperature parameter.

    The temperature scales inputs before softmax, controlling the sharpness
    of the distribution. A higher temperature produces a more uniform
    distribution; a lower temperature produces a sharper one.

    Args:
        start_value: Initial value for the temperature parameter.
    """

    def __init__(self, start_value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.start_value = start_value

    def build(self, input_shape):
        self.temperature = self.add_weight(
            name="temperature",
            shape=(),
            initializer=keras.initializers.Constant(self.start_value),
            trainable=True,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, inputs):
        scaled_inputs = inputs / self.temperature
        return keras.activations.softmax(scaled_inputs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"start_value": self.start_value})
        return config


class DNVS(BaseEstimator):
    """Deep Neural Variable Selection estimator for mean prediction.

    Uses LSTM-based competitive variable selection to simultaneously predict
    a target variable and expose time-varying input variable importance.

    Reference: Aquilina et al. (BIS Working Paper 1291, 2025).

    Args:
        latent_dim: Number of units in the main LSTM layer.
        varsel_dim: Number of units in the variable selection encoding LSTM.
        dropout: Dropout rate for LSTM layers.
        start_temp: Initial temperature for the softmax competition layer.
        gauss_noise: Standard deviation of Gaussian noise applied to inputs
            during training. Set to 0 or None to disable.
        learning_rate: Learning rate for the Adam optimizer.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        varsel_dim: int = 8,
        dropout: float = 0.5,
        start_temp: float = 1.0,
        gauss_noise: float = 0.01,
        learning_rate: float = 0.001,
    ):
        self.latent_dim = latent_dim
        self.varsel_dim = varsel_dim
        self.dropout = dropout
        self.start_temp = start_temp
        self.gauss_noise = gauss_noise
        self.learning_rate = learning_rate

    def _create_NN(self):
        inputs = L.Input(shape=(None, self.num_inputs_), name="Inputs")
        if self.gauss_noise:
            x = L.GaussianNoise(self.gauss_noise, name="GaussianNoise")(inputs)
        else:
            x = inputs
        var_sel = L.LSTM(
            self.varsel_dim,
            return_sequences=True,
            dropout=self.dropout,
            name="InputEncodingForWeights",
        )(x)
        var_sel = L.LSTM(
            self.num_inputs_,
            return_sequences=True,
            dropout=self.dropout,
            name="WeightLogits",
        )(var_sel)
        var_sel = TemperatureScaledSoftmax(
            start_value=self.start_temp, name="CompetitiveSelection"
        )(var_sel)
        weighted_features = L.Multiply(name="Multiply")([x, var_sel])
        encoded = L.LSTM(
            self.latent_dim,
            return_sequences=True,
            dropout=self.dropout,
            name="MainLSTM",
        )(weighted_features)
        output = L.LSTM(
            1,
            activation="linear",
            return_sequences=True,
            dropout=self.dropout,
            name="Output",
        )(encoded)

        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        fit_args: dict | None = None,
    ):
        """Fit the DNVS model.

        Args:
            X: Input array of shape (samples, timesteps, features).
            y: Target array of shape (samples, timesteps, 1).
            validation_data: Optional tuple (X_val, y_val) for early stopping.
            fit_args: Dict of arguments passed to keras Model.fit().
                Defaults to 300 epochs with early stopping and LR reduction.

        Returns:
            self
        """
        if fit_args is None:
            fit_args = {
                "epochs": 300,
                "batch_size": 10,
                "shuffle": True,
                "verbose": 1,
                "callbacks": [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=15, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.9, patience=5, min_lr=0.0001
                    ),
                ],
            }
        self.num_inputs_ = X.shape[-1]
        self.model_ = self._create_NN()
        self.history_ = self.model_.fit(
            X, y, validation_data=validation_data, **fit_args
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted DNVS model.

        Args:
            X: Input array of shape (samples, timesteps, features).

        Returns:
            Predictions of shape (samples, timesteps, 1).
        """
        return self.model_.predict(X)

    def get_variable_weights(self, X: np.ndarray) -> np.ndarray:
        """Get time-varying variable importance weights.

        Args:
            X: Input array of shape (samples, timesteps, features).

        Returns:
            Array of shape (samples, timesteps, features) with weights
            summing to 1 along the features axis.
        """
        varsel_model = keras.Model(
            inputs=self.model_.input,
            outputs=self.model_.get_layer("CompetitiveSelection").output,
        )
        return varsel_model.predict(X)

    def get_learned_temperature(self) -> np.ndarray:
        """Get the learned temperature parameter value.

        Returns:
            The temperature value from the CompetitiveSelection layer.
        """
        return np.array(self.model_.get_layer("CompetitiveSelection").temperature)


class DNVS_Q(BaseEstimator):
    """Deep Neural Variable Selection estimator for quantile prediction.

    Produces multi-quantile forecasts with separate variable selection
    branches per quantile, allowing quantile-specific feature importance.

    Reference: Aquilina et al. (BIS Working Paper 1291, 2025).

    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
        latent_dim: Number of units in the main LSTM layer per quantile branch.
        varsel_dim: Number of units in the variable selection encoding LSTM.
        dropout: Dropout rate for LSTM layers.
        start_temp: Initial temperature for the softmax competition layers.
        gauss_noise: Standard deviation of Gaussian noise applied to inputs
            during training. Set to 0 or None to disable.
        learning_rate: Learning rate for the Adam optimizer.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        latent_dim: int = 32,
        varsel_dim: int = 8,
        dropout: float = 0.5,
        start_temp: float = 1.0,
        gauss_noise: float = 0.01,
        learning_rate: float = 0.001,
    ):
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        self.latent_dim = latent_dim
        self.varsel_dim = varsel_dim
        self.dropout = dropout
        self.start_temp = start_temp
        self.gauss_noise = gauss_noise
        self.learning_rate = learning_rate

    def _create_NN(self):
        inputs = L.Input(shape=(None, self.num_inputs_), name="Inputs")
        if self.gauss_noise:
            x = L.GaussianNoise(self.gauss_noise, name="GaussianNoise")(inputs)
        else:
            x = inputs
        var_sel = L.LSTM(
            self.varsel_dim,
            return_sequences=True,
            dropout=self.dropout,
            name="InputEncodingForWeights",
        )(x)

        quantile_outputs = []
        for quant in self.quantiles:
            q = int(quant * 100)
            var_sel_q = L.LSTM(
                self.num_inputs_,
                return_sequences=True,
                dropout=self.dropout,
                name=f"WeightLogits_q{q}",
            )(var_sel)
            var_sel_q = TemperatureScaledSoftmax(
                start_value=self.start_temp, name=f"CompetitiveSelection_q{q}"
            )(var_sel_q)
            weighted_features_q = L.Multiply(name=f"Multiply_q{q}")([x, var_sel_q])
            encoded_q = L.LSTM(
                self.latent_dim,
                return_sequences=True,
                dropout=self.dropout,
                name=f"MainLSTM_q{q}",
            )(weighted_features_q)
            output_q = L.LSTM(
                1,
                activation="linear",
                return_sequences=True,
                dropout=self.dropout,
                name=f"Output_q{q}",
            )(encoded_q)
            quantile_outputs.append(output_q)

        model = keras.Model(inputs=inputs, outputs=quantile_outputs)

        loss_dict = {
            f"Output_q{int(q * 100)}": quantile_loss(q) for q in self.quantiles
        }
        metrics_dict = {
            f"Output_q{int(q * 100)}": keras.metrics.RootMeanSquaredError()
            for q in self.quantiles
        }

        model.compile(
            loss=loss_dict,
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=metrics_dict,
        )
        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        fit_args: dict | None = None,
    ):
        """Fit the quantile DNVS model.

        Args:
            X: Input array of shape (samples, timesteps, features).
            y: Target array of shape (samples, timesteps, 1). The same target
                is used for all quantile branches.
            validation_data: Optional tuple (X_val, y_val) for early stopping.
            fit_args: Dict of arguments passed to keras Model.fit().
                Defaults to 300 epochs with early stopping and LR reduction.

        Returns:
            self
        """
        if fit_args is None:
            fit_args = {
                "epochs": 300,
                "batch_size": 10,
                "shuffle": True,
                "verbose": 1,
                "callbacks": [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=15, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.9, patience=5, min_lr=0.0001
                    ),
                ],
            }
        self.num_inputs_ = X.shape[-1]
        self.model_ = self._create_NN()
        # Replicate y for each quantile output
        y_targets = [y] * len(self.quantiles)
        val = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val = (X_val, [y_val] * len(self.quantiles))
        self.history_ = self.model_.fit(X, y_targets, validation_data=val, **fit_args)
        return self

    def predict(self, X: np.ndarray) -> list[np.ndarray]:
        """Predict quantiles using the fitted DNVS_Q model.

        Args:
            X: Input array of shape (samples, timesteps, features).

        Returns:
            List of predictions, one per quantile, each of shape
            (samples, timesteps, 1).
        """
        return self.model_.predict(X)

    def get_variable_weights(
        self, X: np.ndarray, quantile: float | None = None
    ) -> np.ndarray:
        """Get time-varying variable importance weights.

        Args:
            X: Input array of shape (samples, timesteps, features).
            quantile: Which quantile branch to extract weights from.
                If None, uses the first quantile.

        Returns:
            Array of shape (samples, timesteps, features) with weights
            summing to 1 along the features axis.
        """
        if quantile is None:
            quantile = self.quantiles[0]
        q = int(quantile * 100)
        layer_name = f"CompetitiveSelection_q{q}"
        varsel_model = keras.Model(
            inputs=self.model_.input,
            outputs=self.model_.get_layer(layer_name).output,
        )
        return varsel_model.predict(X)

    def get_learned_temperature(self, quantile: float | None = None) -> np.ndarray:
        """Get the learned temperature parameter for a quantile branch.

        Args:
            quantile: Which quantile branch. If None, uses the first quantile.

        Returns:
            The temperature value from the CompetitiveSelection layer.
        """
        if quantile is None:
            quantile = self.quantiles[0]
        q = int(quantile * 100)
        layer_name = f"CompetitiveSelection_q{q}"
        return np.array(self.model_.get_layer(layer_name).temperature)
