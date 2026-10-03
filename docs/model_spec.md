# Frozen UGRF v1 model specification

UGRF combines a TSB baseline and a pooled-age Renewal candidate. For each SKU `i` and historical forecast origin `o`, both experts are fitted using only data available through that origin and scored on the same cumulative forecast horizon.

## TSB

The frozen reference uses `alpha = beta = 0.10`.

Occurrence probability is initialized with a half-count correction,

`p0 = (number_of_positive_periods + 0.5) / (n_periods + 1)`,

and positive size is initialized at the mean positive demand. The implementation then applies the frozen recursive update order over the full training history.

## Renewal occurrence model

The pooled age shape is

`logit P(O_it = 1) = alpha_i + beta1 log(A_it) + beta2 log(A_it)^2`.

`beta1` and `beta2` are estimated from completed inter-demand intervals pooled across SKUs using the discrete-time risk set and a ridge penalty of 1 on each age coefficient. The pooled shape is then held fixed while a local intercept `alpha_i` is fitted to each SKU's occurrence history.

## Positive magnitude

The panel anchor is the intercept from a pooled log-link count model with `log(age)` as a covariate and ridge penalty 1 on the age coefficient. The local mean is computed on log positive demand and shrunk toward the panel anchor with pseudo-count strength `k=5`.

## Multi-step forecast

Future probability mass is propagated across age states. Event mass resets to age 1; no-event mass advances to age + 1. Expected lead demand equals marginal event probability times the fitted deterministic positive magnitude.

## Utility gate

At each historical origin:

`G_i,o = L_i,o(TSB) - L_i,o(Renewal)`

with cumulative absolute error as the frozen loss. Historical utility is

`U_i = sum_o G_i,o`.

The final decision is strictly:

`Renewal if U_i > 0 else TSB`.

No threshold tuning, recency weighting, confidence filter, or dataset identifier is part of UGRF v1.
