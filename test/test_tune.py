import unittest
import os
import pandas as pd

class TestTuneModel(unittest.TestCase):
    def test_reg_xgb(self):
        path = os.path.join("data", "pmsm_temperature_data.csv")
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        num_targets = ['pm', 'stator_winding']
        cat_targets = ['stator_yoke'] 
        X = df.drop(num_targets+cat_targets, axis=1)
        y = df[num_targets+cat_targets]
