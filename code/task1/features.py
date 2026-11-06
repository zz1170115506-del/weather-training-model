"""Fold-local transformations, with original-feature group provenance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import NUMERIC, CATEGORICAL, FEATURES


class Features:
    def __init__(self, options=None, native_categories=False, scale=False):
        self.options = dict(options or {})
        self.native_categories = native_categories
        self.scale = scale
        if self.options.get("percentage_policy", "retain") not in ("retain", "clip"):
            raise ValueError("percentage_policy must be retain or clip.")

    def _derive(self, X):
        frame = X[FEATURES].copy()
        self._derived_sources = {}
        for col in ("Humidity", "Precipitation (%)"):
            if self.options.get("out_of_range_flags", False):
                flag = f"{col}__outside"
                frame[flag] = ((frame[col] < 0) | (frame[col] > 100)).astype(float)
                self._derived_sources[flag] = col
            if self.options.get("percentage_policy", "retain") == "clip":
                frame[col] = frame[col].clip(0, 100)
        if self.options.get("interactions", False):
            pairs = [("Temperature", "Humidity"), ("Temperature", "Precipitation (%)"),
                     ("Atmospheric Pressure", "Precipitation (%)"), ("Visibility (km)", "Humidity")]
            for a, b in pairs:
                name = f"{a}__times__{b}"
                frame[name] = frame[a] * frame[b]
                self._derived_sources[name] = name
        return frame

    def fit(self, X):
        frame = self._derive(X)
        self.numeric_ = [c for c in frame if c not in CATEGORICAL]
        if self.native_categories:
            self.medians_ = frame[self.numeric_].median().fillna(0)
            self.columns_ = list(frame.columns)
            self.groups_ = np.asarray([self._derived_sources.get(c, c) for c in self.columns_], dtype=object)
            self.feature_names_ = list(self.columns_)
            if self.native_categories == "hgb":
                self.category_levels_ = {col: sorted(frame[col].dropna().astype(str).unique())
                                         for col in CATEGORICAL}
        else:
            numeric = [("fill", SimpleImputer(strategy="median", keep_empty_features=True))]
            if self.scale:
                numeric.append(("scale", StandardScaler()))
            self.transformer_ = ColumnTransformer([
                ("numeric", Pipeline(numeric), self.numeric_),
                ("categorical", Pipeline([
                    ("fill", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), CATEGORICAL),
            ])
            self.transformer_.fit(frame)
            encoder = self.transformer_.named_transformers_["categorical"].named_steps["encode"]
            sources = [self._derived_sources.get(c, c) for c in self.numeric_]
            for col, categories in zip(CATEGORICAL, encoder.categories_):
                sources.extend([col] * len(categories))
            self.groups_ = np.asarray(sources, dtype=object)
            self.feature_names_ = self.transformer_.get_feature_names_out().tolist()
        self.fitted_rows_ = len(X)
        return self

    def transform(self, X):
        if not hasattr(self, "groups_"):
            raise RuntimeError("Fit features before transform.")
        frame = self._derive(X)
        if self.native_categories:
            frame[self.numeric_] = frame[self.numeric_].fillna(self.medians_)
            for col in CATEGORICAL:
                if self.native_categories == "hgb":
                    frame[col] = pd.Categorical(frame[col].astype("string"), categories=self.category_levels_[col])
                else:
                    frame[col] = frame[col].fillna("__MISSING__").astype(str)
            return frame[self.columns_]
        values = np.asarray(self.transformer_.transform(frame), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Feature transformation produced nonfinite values.")
        return values

    def fit_transform(self, X):
        return self.fit(X).transform(X)
