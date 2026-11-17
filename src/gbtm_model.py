"""Dependency-light implementation of the project's polynomial GBTM."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PureGBTM:
    """EM-style Gaussian polynomial trajectory groups fitted by weighted LS."""

    def __init__(self, n_groups=3, degree=2, max_iter=100, tol=1e-4):
        self.n_groups = n_groups
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.group_params = []
        self.sigmas = []
        self.pi = None

    def _design(self, timepoints):
        t = np.asarray(timepoints, dtype=float)
        return np.column_stack([t ** degree for degree in range(self.degree + 1)])

    @staticmethod
    def _log_likelihood(X, y, beta, sigma, proportion):
        residual = y - X @ beta
        return np.log(proportion + 1e-12) - np.sum(
            0.5 * np.log(2 * np.pi * sigma**2) + residual**2 / (2 * sigma**2)
        )

    def fit(self, frame, id_col, time_col, val_col):
        subjects = frame[id_col].drop_duplicates().to_numpy()
        subject_data = {}
        for subject in subjects:
            sub = frame.loc[frame[id_col] == subject].sort_values(time_col)
            subject_data[subject] = (
                self._design(sub[time_col].to_numpy()),
                sub[val_col].to_numpy(dtype=float),
            )
        np.random.seed(42)
        responsibilities = np.random.dirichlet(np.ones(self.n_groups), size=len(subjects))
        self.pi = np.full(self.n_groups, 1.0 / self.n_groups)

        for _ in range(self.max_iter):
            old_pi = self.pi.copy()
            self.group_params, self.sigmas = [], []
            for group in range(self.n_groups):
                xs, ys, ws = [], [], []
                for index, subject in enumerate(subjects):
                    X_s, y_s = subject_data[subject]
                    xs.append(X_s)
                    ys.append(y_s)
                    ws.append(np.full(len(y_s), responsibilities[index, group]))
                X_all, y_all, weights = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
                root_w = np.sqrt(weights + 1e-6)
                beta = np.linalg.lstsq(X_all * root_w[:, None], y_all * root_w, rcond=None)[0]
                residual = y_all - X_all @ beta
                sigma = np.sqrt(np.sum(weights * residual**2) / np.sum(weights) + 1e-6)
                self.group_params.append(beta)
                self.sigmas.append(sigma)
            self.pi = responsibilities.mean(axis=0)

            log_likelihood = np.zeros((len(subjects), self.n_groups))
            for index, subject in enumerate(subjects):
                X_s, y_s = subject_data[subject]
                for group in range(self.n_groups):
                    log_likelihood[index, group] = self._log_likelihood(
                        X_s, y_s, self.group_params[group], self.sigmas[group], self.pi[group]
                    )
            max_log = log_likelihood.max(axis=1, keepdims=True)
            likelihood = np.exp(log_likelihood - max_log)
            responsibilities = likelihood / likelihood.sum(axis=1, keepdims=True)
            if np.max(np.abs(self.pi - old_pi)) < self.tol:
                break

        assigned = np.argmax(responsibilities, axis=1)
        result = pd.DataFrame({
            id_col: subjects,
            "gbtm_group": assigned,
            "group_prob": responsibilities.max(axis=1),
        })
        return result, responsibilities

    def predict_groups(self, frame, id_col, time_col, val_col):
        if not self.group_params:
            raise RuntimeError("Fit GBTM before predicting groups.")
        subjects = frame[id_col].drop_duplicates().to_numpy()
        likelihood = np.zeros((len(subjects), self.n_groups))
        for index, subject in enumerate(subjects):
            sub = frame.loc[frame[id_col] == subject].sort_values(time_col)
            X_s = self._design(sub[time_col].to_numpy())
            y_s = sub[val_col].to_numpy(dtype=float)
            for group in range(self.n_groups):
                likelihood[index, group] = self._log_likelihood(
                    X_s, y_s, self.group_params[group], self.sigmas[group], self.pi[group]
                )
        max_log = likelihood.max(axis=1, keepdims=True)
        probabilities = np.exp(likelihood - max_log)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return pd.DataFrame({
            id_col: subjects,
            "gbtm_group": probabilities.argmax(axis=1),
            "group_prob": probabilities.max(axis=1),
        })
