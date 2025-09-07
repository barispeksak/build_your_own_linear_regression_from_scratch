
"""
    Linear Regression implementation from scratch with multiple features support,
    Regularization, Adam optimizer, and Early stopping.

    Author: Barış Peksak
"""

import pandas as pd
import numpy as np
import math
import copy


class LinearRegression:
    

    def __init__(self):
        
        # Attributes

        self.w = None
        self.b = None
        self.is_fitted = False
        self.mean = None
        self.std = None
        self.standardize = False
        self.early_stopping = False
        self.regularization = None
        self.lambda_reg = 0.01
        self.optimizer = None

    def compute_cost(self, X, y, w, b):
        """Calculate the cost function."""
        number_of_samples = X.shape[0]

        # Vectorized computation
        f_wb = X @ w + b
        cost = np.sum((f_wb - y)**2) / (2 * number_of_samples)

        # Added Ridge regularization 
        if self.regularization == 'ridge':
            cost += np.sum(w**2) * (self.lambda_reg / (2 * number_of_samples))

        return cost

    def compute_gradient(self, X, y, w, b):
        """Calculate gradients."""
        number_of_samples = X.shape[0]

        # Vectorized computation 
        f_wb = X @ w + b

        dj_dw = X.T @ (f_wb - y) / number_of_samples
        dj_db = np.sum((f_wb - y)) / number_of_samples
        
        # Added Ridge regularization to prevent the converge on 0 for gradients.
        if self.regularization == 'ridge':
            dj_dw += (self.lambda_reg * w) / number_of_samples

        return dj_dw, dj_db

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters, tolerance=1e-6, patience=10):
        """Standard gradient descent algorithm."""
        number_of_samples = X.shape[0]
        J_history = []
        w_history = []
        w = copy.deepcopy(w_in)
        b = b_in

        for i in range(num_iters):
            
            dj_dw, dj_db = self.compute_gradient(X, y, w, b)
            w = w - (alpha * dj_dw)
            b = b - (alpha * dj_db)

            # Track cost for first 10,000.
            if i < 10000:
                cost = self.compute_cost(X, y, w, b)
                J_history.append(cost)

            # Print every 10% of iterations
            if i % math.ceil(num_iters/10) == 0:
                w_history.append(w)
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

            # Cheking Early stopping.
            if self.early_stopping and len(J_history) > patience:
                recent_improvement = J_history[-patience-1] - J_history[-1]
                if recent_improvement < tolerance:
                    print(f"Converged early at iteration {i}")
                    break

        return w, b, J_history, w_history

    def adam_optimizer(self, X, y, w_in, b_in, alpha, num_iters, tolerance, patience,
                       beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimization algorithm."""
        number_of_samples = X.shape[0]
        
        
        w = copy.deepcopy(w_in)
        b = b_in
        m_w = np.zeros_like(w)  
        v_w = np.zeros_like(w)  
        m_b = 0                 
        v_b = 0                 
        
        J_history = []
        w_history = []

        for i in range(num_iters):
            
            dj_dw, dj_db = self.compute_gradient(X, y, w, b)

            # Update momentum and velocity
            m_w = beta1 * m_w + (1 - beta1) * dj_dw
            v_w = beta2 * v_w + (1 - beta2) * (dj_dw ** 2)
            m_b = beta1 * m_b + (1 - beta1) * dj_db
            v_b = beta2 * v_b + (1 - beta2) * (dj_db ** 2)

            # Correcting Bias
            m_w_corrected = m_w / (1 - beta1 ** (i + 1))
            v_w_corrected = v_w / (1 - beta2 ** (i + 1))
            m_b_corrected = m_b / (1 - beta1 ** (i + 1))
            v_b_corrected = v_b / (1 - beta2 ** (i + 1))

            # Update parameters
            w = w - alpha * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)
            b = b - alpha * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)

            # Track cost.
            if i < 10000:
                cost = self.compute_cost(X, y, w, b)
                J_history.append(cost)

            if i % math.ceil(num_iters/10) == 0:
                w_history.append(w)
                print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

            if self.early_stopping and len(J_history) > patience:
                recent_improvement = J_history[-patience-1] - J_history[-1]
                if recent_improvement < tolerance:
                    print(f"Converged early at iteration {i}")
                    break

        return w, b, J_history, w_history

    def fit(self, X, y, w_in=None, b_in=0, alpha=0.01, num_iters=1000,
            standardize=False, early_stopping=False, tolerance=1e-6, patience=10,
            regularization=None, lambda_reg=0.01, optimizer=None):
        """
        Train the linear regression model.
        """
        if w_in is None:
            n_features = X.shape[1] if X.ndim > 1 else 1
            w_in = np.zeros(n_features)

        
        self.standardize = standardize
        self.early_stopping = early_stopping
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.optimizer = optimizer

        # Apply standardization
        if standardize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X_norm = (X - self.mean) / self.std
            X_to_use = X_norm
        else:
            X_to_use = X

        # Optimizer
        if self.optimizer == 'adam':
            w, b, _, _ = self.adam_optimizer(X_to_use, y, w_in, b_in, alpha, num_iters, tolerance, patience)
        else:
            w, b, _, _ = self.gradient_descent(X_to_use, y, w_in, b_in, alpha, num_iters, tolerance, patience)

    
        self.w = w
        self.b = b
        self.is_fitted = True

    def predict(self, X):
        """
        Make predictions on data.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")

        if self.standardize:
            X_to_use = (X - self.mean) / self.std
        else:
            X_to_use = X

        return (X_to_use @ self.w) + self.b