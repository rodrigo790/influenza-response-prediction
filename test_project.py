import json
import unittest

import pandas as pd

from analysis import model_set
from common import RESULTS


class PredictionTests(unittest.TestCase):
    def test_selected_features_are_inside_each_estimator(self):
        models = model_set()
        for name in ("Expression ridge", "Expression random forest"):
            _, pipeline = models[name]
            self.assertEqual(pipeline.steps[0][0], "selectkbest")
            self.assertEqual(pipeline.steps[0][1].k, 30)

    def test_partition_and_metric_coverage(self):
        summary = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(summary["participants"], summary["holdout_n"] + summary["development_n"])
        metrics = pd.read_csv(RESULTS / "metrics.csv")
        self.assertEqual(set(metrics.model), set(model_set()))
        self.assertTrue(metrics.groupby("model").partition.value_counts().unstack().holdout.eq(1).all())
        self.assertTrue(metrics.groupby("model").partition.value_counts().unstack().development_cv.eq(5).all())
