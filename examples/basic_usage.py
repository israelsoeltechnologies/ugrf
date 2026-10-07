import numpy as np

from ugrf import RenewalForecaster, TSB, UtilityGatedRenewal


panel = np.array(
    [
        [0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 5, 0, 0, 7, 0, 0, 6],
        [0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0],
        [1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 2, 0, 0],
    ],
    dtype=float,
)

print("TSB cumulative:", TSB().fit(panel[0, :15]).predict_cumulative(3))

renewal = RenewalForecaster().fit(panel, origin=15)
print("Renewal point paths:\n", renewal.predict(horizon=3))
print("Renewal cumulative:", renewal.cumulative_for_training_panel(horizon=3))

ugrf = UtilityGatedRenewal(horizon=3).fit(
    panel,
    utility_origins=[9, 12],
    final_origin=15,
)

# Standard forecasting API: one point forecast per lead.
print("UGRF point paths:\n", ugrf.predict())

# Operational cumulative target used in the Paper-1 benchmark.
print("UGRF cumulative:", ugrf.predict_cumulative())

print("utility:", ugrf.utility_)
print("selected:", ugrf.selected_model_)
