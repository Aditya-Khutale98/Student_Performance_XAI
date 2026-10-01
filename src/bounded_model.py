import numpy as np
from sklearn.linear_model import LinearRegression


class BoundedRegression:
    """
    Regression model whose predictions are naturally bounded
    between 0 and 100 using a logistic transformation.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):

        # Convert target from 0-100 to 0-1
        y_scaled = np.clip(
            np.asarray(y) / 100.0,
            0.001,
            0.999
        )

        # Logit transformation
        y_logit = np.log(
            y_scaled / (1 - y_scaled)
        )

        self.model.fit(
            X,
            y_logit
        )

        return self

    def predict(self, X):

        # Prediction in logit space
        z = self.model.predict(X)

        # Sigmoid transformation
        probability = 1 / (
            1 + np.exp(-z)
        )

        # Convert back to 0-100
        prediction = probability * 100

        # Final safety bound
        return np.clip(
            prediction,
            0,
            100
        )