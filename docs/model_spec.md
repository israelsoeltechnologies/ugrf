# Frozen UGRF v1 Model Specification

UGRF combines a TSB baseline and a pooled-age Renewal candidate. For each SKU \(i\) and historical forecast origin \(o\), both experts are fitted using only information available at that origin and scored on the same cumulative forecast horizon.

## TSB baseline

The frozen reference uses:

\[
\alpha=\beta=0.10.
\]

Occurrence probability is initialized with a half-count correction:

\[
p_0=\frac{n_{positive}+0.5}{n_{periods}+1}.
\]

Positive size is initialized at the mean positive demand. The package then applies the frozen recursive update order over the complete training history.

## Renewal occurrence model

\[
\operatorname{logit}[P(O_{i,t}=1)]
=
\alpha_i+\beta_1\log A_{i,t}+\beta_2(\log A_{i,t})^2.
\]

The pooled age-shape coefficients are estimated from completed inter-demand intervals across the panel using the discrete-time risk set. A ridge penalty of 1 is applied to each age coefficient.

The pooled shape is then held fixed while the SKU-specific occurrence intercept \(\alpha_i\) is fitted locally.

## Positive magnitude

The panel anchor is the intercept from a pooled log-link count model with `log(age)` as a covariate and ridge penalty 1 on the age coefficient.

The SKU-level local log-positive mean is shrunk toward the panel anchor with pseudo-count strength:

\[
k=5.
\]

## Multi-step Renewal forecast

Future probability mass is propagated across demand-age states. Event mass resets to age 1; no-event mass advances to age \(a+1\).

If \(q_{i,h}\) is the marginal event probability at lead \(h\) and \(\hat\mu_i\) is the fitted positive magnitude:

\[
\hat y_{i,t+h}^{R}=\hat\mu_i q_{i,h}.
\]

The cumulative Renewal forecast is:

\[
\hat D_i^{(H),R}=\sum_{h=1}^{H}\hat y_{i,t+h}^{R}.
\]

## Historical utility

At historical origin \(o\):

\[
G_{i,o}=L_{i,o}(\mathrm{TSB})-L_{i,o}(R).
\]

Historical utility is:

\[
U_i=\sum_{o\in\mathcal O_i}G_{i,o}.
\]

## Utility gate

\[
w_i=\mathbb I(U_i>0).
\]

The final lead-by-lead point forecast is:

\[
\hat{\mathbf y}^{UGRF}_i
=
w_i\hat{\mathbf y}^{R}_i+(1-w_i)\hat{\mathbf y}^{TSB}_i.
\]

The cumulative forecast is:

\[
\hat D_i^{(H),UGRF}=\sum_{h=1}^{H}\hat y_{i,t+h}^{UGRF}.
\]

If no valid historical utility evidence exists, utility is set to zero and TSB remains the fallback.

## Frozen v1 boundary

The canonical model contains no dataset-specific threshold, threshold tuning, recency weighting, confidence filter, dataset identifier, or ML routing model.
