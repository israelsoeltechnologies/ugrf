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

print("TSB:", TSB().fit(panel[0, :15]).predict_cumulative(3))

renewal = RenewalForecaster().fit(panel, origin=15)
print("Renewal:", renewal.cumulative_for_training_panel(horizon=3))

ugrf = UtilityGatedRenewal(horizon=3).fit(
    panel,
    utility_origins=[9, 12],
    final_origin=15,
)
print("UGRF:", ugrf.predict_cumulative())
print("utility:", ugrf.utility_)
print("selected:", ugrf.selected_model_)
