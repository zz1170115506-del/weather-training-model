"""Independent posterior and preprocessing checks for contamination histograms."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generative_weather import ContaminationHistogramNB


class GenerativeTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"temperature": [-2., -1., 0., 1., 2., 3.],
                               "cloud": ["a", "a", "b", "b", "c", "c"]})
        self.y = np.array([1, 1, 2, 2, 4, 4])

    def test_plain_categorical_nb_matches_hand_posterior(self):
        X = pd.DataFrame({"cloud": ["a", "a", "b", "b"]})
        model = ContaminationHistogramNB(alpha=1, contamination=0,
                                          categorical_features=["cloud"]).fit(X, [1, 1, 4, 4])
        # Three bins a,b,unknown; class 1 conditional [3,1,1]/5, class 4 [1,3,1]/5.
        np.testing.assert_allclose(model.predict_proba(pd.DataFrame({"cloud": ["a", "new"]})),
                                   [[3 / 4, 1 / 4], [1 / 2, 1 / 2]])
        self.assertEqual(model.n_iter_, 0)
        self.assertEqual(model.contamination_, 0)

    def test_one_em_iteration_matches_responsibility_and_m_step_by_hand(self):
        X = pd.DataFrame({"cloud": ["a", "a", "b", "b"]})
        # Initial clean likelihood at every true label is .6 and shared noise .428571.
        initial_noise = 3 / 7
        r = (0.8 * 0.6) / (0.8 * 0.6 + 0.2 * initial_noise)
        eps = 1 - r
        clean = np.array([2 * r + 1, 1, 1]) / (2 * r + 3)
        noise = np.array([2 * (1 - r) + 1, 2 * (1 - r) + 1, 1]) / (4 * (1 - r) + 3)
        model = ContaminationHistogramNB(alpha=1, contamination=0.2, max_iter=1,
                                          categorical_features=["cloud"]).fit(X, [1, 1, 4, 4])
        self.assertAlmostEqual(model.contamination_, eps)
        np.testing.assert_allclose(model.clean_feature_prob_[0][0], clean)
        np.testing.assert_allclose(model.noise_feature_prob_[0], noise)
        mixture = np.array([(1 - eps) * clean[0] + eps * noise[0],
                            (1 - eps) * clean[1] + eps * noise[0]])
        np.testing.assert_allclose(model.predict_proba(pd.DataFrame({"cloud": ["a"]}))[0], mixture / mixture.sum())

    def test_mixture_indicator_is_shared_across_features(self):
        X = pd.DataFrame({"first": ["a", "a", "b", "b"], "second": ["u", "v", "u", "v"]})
        model = ContaminationHistogramNB(alpha=0.5, contamination=0.1, max_iter=2,
                                          categorical_features=list(X.columns)).fit(X, [1, 1, 4, 4])
        test = pd.DataFrame({"first": ["a"], "second": ["u"]})
        clean = np.array([model.clean_feature_prob_[0][c, 0] * model.clean_feature_prob_[1][c, 0]
                          for c in range(2)])
        noise = model.noise_feature_prob_[0][0] * model.noise_feature_prob_[1][0]
        expected = ((1 - model.contamination_) * clean + model.contamination_ * noise) * model.class_prior_
        np.testing.assert_allclose(model.predict_proba(test)[0], expected / expected.sum())

    def test_new_categories_extremes_missing_and_fixed_class_order_are_finite(self):
        model = ContaminationHistogramNB(categorical_features=["cloud"], class_labels=[4, 3, 2, 1]).fit(self.X, self.y)
        test = pd.DataFrame({"temperature": [-1e100, 1e100, np.nan], "cloud": ["new", None, "a"]})
        probabilities = model.predict_proba(test)
        self.assertEqual(probabilities.shape, (3, 4))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities > 0).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(model.classes_, [4, 3, 2, 1])
        self.assertEqual(model.predict_proba(test.iloc[:0]).shape, (0, 4))

    def test_predict_does_not_change_training_bins_or_categories(self):
        model = ContaminationHistogramNB(n_bins=3, categorical_features=["cloud"]).fit(self.X, self.y)
        edges = model.bin_edges_["temperature"].copy()
        mapping = dict(model.category_maps_["cloud"])
        model.predict_proba(pd.DataFrame({"temperature": [1e9], "cloud": ["unseen"]}))
        np.testing.assert_array_equal(model.bin_edges_["temperature"], edges)
        self.assertEqual(model.category_maps_["cloud"], mapping)
        self.assertEqual(model.n_samples_fit_, len(self.X))
        np.testing.assert_allclose(edges, np.quantile(self.X.temperature, [1 / 3, 2 / 3]))

    def test_deterministic_em_and_parameter_roundtrip(self):
        kwargs = dict(n_bins=3, alpha=0.1, categorical_features=["cloud"], class_labels=[1, 2, 3, 4])
        first = ContaminationHistogramNB(**kwargs).fit(self.X, self.y)
        second = ContaminationHistogramNB(**first.get_params()).fit(self.X, self.y)
        np.testing.assert_array_equal(first.predict_proba(self.X), second.predict_proba(self.X))
        np.testing.assert_array_equal(first.log_likelihood_history_, second.log_likelihood_history_)
        self.assertTrue(0.01 <= first.contamination_ <= 0.4)
        self.assertLessEqual(first.n_iter_, first.max_iter)
        self.assertTrue(np.isfinite(first.log_likelihood_history_).all())

    def test_array_input_and_all_missing_numeric_feature(self):
        X = np.array([[np.nan, "a"], [np.nan, "b"], [np.nan, "a"]], dtype=object)
        model = ContaminationHistogramNB(categorical_features=[1]).fit(X, [1, 4, 1])
        np.testing.assert_allclose(model.predict_proba([[np.nan, "new"]]).sum(axis=1), 1)
        self.assertEqual(len(model.bin_edges_[0]), 0)

    def test_invalid_parameters_labels_schema_and_unfitted_prediction(self):
        invalid = [{"n_bins": 1}, {"n_bins": True}, {"alpha": 0}, {"alpha": np.inf},
                   {"contamination": -0.1}, {"contamination": 0.8}, {"max_iter": 0}, {"tol": -1},
                   {"min_contamination": 0.5, "max_contamination": 0.4},
                   {"categorical_features": "cloud"}, {"categorical_features": ["missing"]},
                   {"class_labels": [1, 1]}, {"class_labels": [1, 2]}, {"class_labels": [1, np.nan]}]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContaminationHistogramNB(**kwargs).fit(self.X, self.y)
        with self.assertRaises(RuntimeError):
            ContaminationHistogramNB().predict([[0]])
        model = ContaminationHistogramNB(categorical_features=["cloud"]).fit(self.X, self.y)
        for test in [self.X[["cloud", "temperature"]], self.X[["cloud"]],
                     pd.DataFrame({"temperature": [np.inf], "cloud": ["a"]})]:
            with self.assertRaises(ValueError):
                model.predict_proba(test)
        for X, y in [(np.empty((0, 2)), []), ([[0], [1]], [1]), ([[0], [1]], [1, np.nan])]:
            with self.assertRaises(ValueError):
                ContaminationHistogramNB().fit(X, y)


if __name__ == "__main__":
    unittest.main()
