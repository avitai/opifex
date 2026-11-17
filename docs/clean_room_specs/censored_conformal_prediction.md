# Mathematical Specification: Right-Censored & Doubly-Robust Conformal Prediction

This specification records the peer-reviewed mathematical equations, statistical assumptions, and boundary invariants implemented in `opifex.uncertainty.conformal` for distribution-free lower prediction bounds under right censoring.

## Primary References

1. **Candès, E. J., Lei, L., & Ren, Z. (2023).** *Conformalized Survival Analysis*. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 85(1), 24–45 (`arXiv:2103.09763`).
2. **Tibshirani, R. J., Foygel Barber, R., Candès, E. J., & Ramdas, A. (2019).** *Conformal Prediction Under Covariate Shift*. *Advances in Neural Information Processing Systems (NeurIPS)*, 32 (`arXiv:1904.06019`).
3. **Sesia, M., & Svetnik, V. (2025).** *Doubly Robust Conformalized Survival Analysis with Right-Censored Data*. *Proceedings of the 42nd International Conference on Machine Learning (ICML)*, PMLR 267 (`arXiv:2412.09729`).

---

## 1. Statistical Model & Censoring Filtration

Let $(X_i, T_i, C_i)_{i=1}^{n+1}$ be exchangeable draws from a joint distribution $P_{X, T, C}$ on $\mathcal{X} \times \mathbb{R} \times (\mathbb{R} \cup \{+\infty\})$, where $i \in \{1, \dots, n\}$ indexes the calibration sample and $i = n + 1$ indexes a test draw:

- $X_i \in \mathcal{X}$: covariate vector,
- $T_i \in \mathbb{R}$: latent target (event time, logarithmic potency $\mathrm{pChEMBL}$, or selectivity margin $\Delta\mathrm{p}K_i$),
- $C_i \in \mathbb{R} \cup \{+\infty\}$: right-censoring threshold,
- $\tilde{T}_i := \min(T_i, C_i)$ and $\delta_i := \mathbb{I}(T_i \le C_i)$: observed right-censored outcome and event indicator.

### Assumption 1 (Conditionally Ignorable Censoring & Positivity)

$$T_i \perp\!\!\!\perp C_i \mid X_i, \qquad G(t \mid x) := \Pr(C_i \ge t \mid X_i = x) \ge \varepsilon_{\mathrm{pos}} > 0.$$

### Regime I: Observed Censoring Thresholds (Candès, Lei & Ren 2023, Theorem 1)

Fix a deterministic cutoff $c_0 \in \mathbb{R}$ chosen independently of the calibration set, and define the filtered calibration index set:

$$\mathcal{I}_{c_0} := \big\{i \in \{1, \dots, n\} : C_i \ge c_0\big\}.$$

For every $i \in \mathcal{I}_{c_0}$, the $c_0$-truncated latent outcome is observed without error regardless of $\delta_i$:

$$\min(\tilde{T}_i, c_0) = \min\big(\min(T_i, C_i), c_0\big) = \min(T_i, c_0).$$

By Assumption 1, conditioning on $C_i \ge c_0$ leaves the conditional outcome law $P_{T \mid X}$ invariant and induces a pure covariate shift between $P_{X \mid C \ge c_0}$ and $P_X$ with Radon–Nikodym weight proportional to $1 / G(c_0 \mid x)$:

$$\frac{\mathrm{d}P_X}{\mathrm{d}P_{X \mid C \ge c_0}}(x) = \frac{\Pr(C \ge c_0)}{G(c_0 \mid x)} \propto \frac{1}{G(c_0 \mid x)}.$$

### Regime II: Partially Observed Censoring Thresholds (General Right-Censoring)

When only $(X_i, \tilde{T}_i, \delta_i)$ are observed, $\min(T_i, c_0) = \min(\tilde{T}_i, c_0)$ is observable on $\Omega_{c_0} := \{i : \delta_i = 1 \text{ or } \tilde{T}_i \ge c_0\} = \{i : C_i \ge \min(T_i, c_0)\}$, with Horvitz–Thompson inverse probability of censoring weight $w_i^{\mathrm{HT}} = \mathbb{I}(i \in \Omega_{c_0}) / \hat{G}(\min(\tilde{T}_i, c_0) \mid X_i)$ where $\hat{G}$ is estimated on $(X_j, \tilde{T}_j, 1 - \delta_j)$ (e.g., via the Kaplan–Meier estimator under marginal censoring or a conditional survival model). Alternatively, drawing $\hat{C}_i \sim \hat{G}(\cdot \mid X_i) \mid C_i \ge \tilde{T}_i$ on $\{\delta_i = 1\}$ and setting $\hat{C}_i = \tilde{T}_i$ on $\{\delta_i = 0\}$ reduces general right-censoring to Regime I with covariate-only weights (Sesia & Svetnik 2025).

---

## 2. One-Sided Nonconformity Scores & Lower-Bound Lifting

Given miscoverage level $\alpha \in (0, 1)$ and cutoff $c_0$, one-sided nonconformity scores penalize outcomes falling below the predictor:

1. **Signed Residual Score** (mean/point predictor $\hat{\mu}$):
   $$V_i = \hat{\mu}(X_i) - \min(\tilde{T}_i, c_0), \qquad \hat{L}(x) = \min\!\big(\hat{\mu}(x) - \hat{q}_{1-\alpha}(x),\, c_0\big).$$

2. **Conformalized Lower-Quantile Score** (conditional $\alpha$-quantile predictor $\hat{q}_\alpha$):
   $$V_i = \min\big(\hat{q}_\alpha(X_i), c_0\big) - \min(\tilde{T}_i, c_0), \qquad \hat{L}(x) = \min\!\big(\min(\hat{q}_\alpha(x), c_0) - \hat{q}_{1-\alpha}(x),\, c_0\big).$$

3. **Locally Scaled Signed Residual Score** (predictor $\hat{\mu}$ and positive scale $\hat{\sigma}_+(x) = \max(\hat{\sigma}(x), \sigma_{\min})$):
   $$V_i = \frac{\hat{\mu}(X_i) - \min(\tilde{T}_i, c_0)}{\hat{\sigma}_+(X_i)}, \qquad \hat{L}(x) = \min\!\big(\hat{\mu}(x) - \hat{q}_{1-\alpha}(x)\,\hat{\sigma}_+(x),\, c_0\big).$$

### Monotonicity Lifting to the Latent Target

Because $\hat{L}(x) \le c_0$ and $T_{n+1} \ge \min(T_{n+1}, c_0)$ pointwise,

$$\big\{T_{n+1} \ge \hat{L}(X_{n+1})\big\} = \big\{\min(T_{n+1}, c_0) \ge \hat{L}(X_{n+1})\big\} \supseteq \big\{\min(T_{n+1}, c_0) \ge \hat{L}_{\mathrm{raw}}(X_{n+1})\big\},$$

so any $(1 - \alpha)$ lower prediction bound for $\min(T_{n+1}, c_0)$ is automatically a valid $(1 - \alpha)$ lower prediction bound `[lower, +inf)` for the uncensored latent target $T_{n+1}$:

$$\Pr\!\big(T_{n+1} \ge \hat{L}(X_{n+1})\big) \ge 1 - \alpha.$$

### Doubly-Robust Intersection (Sesia & Svetnik 2025)

When $\hat{q}_\alpha(x)$ estimates the conditional $\alpha$-quantile of $T \mid X = x$, intersecting $\hat{L}_{\mathrm{cal}}(x)$ with $\hat{q}_\alpha(x)$ replaces $\hat{q}_{1-\alpha}(x)$ by $\max(\hat{q}_{1-\alpha}(x), 0)$:

$$\hat{L}_{\mathrm{DR}}(x) := \min\!\big(\min(\hat{q}_\alpha(x), c_0) - \max\big(\hat{q}_{1-\alpha}(x), 0\big),\, c_0\big),$$

preserving asymptotic $(1 - \alpha)$ coverage whenever **either** the censoring model $\hat{G}$ **or** the conditional quantile model $\hat{q}_\alpha$ is consistent.

---

## 3. Weighted Conformal Quantile & Boundary Invariants

For positivity floor $\varepsilon_{\mathrm{pos}} > 0$, define guarded survival $\hat{G}_\varepsilon(c_0 \mid z) := \max(\hat{G}(c_0 \mid z), \varepsilon_{\mathrm{pos}})$ and unnormalized weights:

$$w_i := \frac{\mathbb{I}(C_i \ge c_0)}{\hat{G}_\varepsilon(c_0 \mid X_i)} \;\; (i = 1, \dots, n), \qquad w_{n+1}(x) := \frac{1}{\hat{G}_\varepsilon(c_0 \mid x)}.$$

Normalize over all $n + 1$ atoms with $W_{\mathrm{tot}}(x) := \sum_{j=1}^n w_j + w_{n+1}(x) > 0$:

$$p_i(x) := \frac{w_i}{W_{\mathrm{tot}}(x)} \;\; (i = 1, \dots, n), \qquad p_\infty(x) := \frac{w_{n+1}(x)}{W_{\mathrm{tot}}(x)}.$$

Mapping excluded calibration points ($w_i = 0$) to $\tilde{V}_i = +\infty$ alongside $\tilde{V}_{n+1} = +\infty$, the weighted conformal quantile (Tibshirani et al. 2019, Eq. 5) is:

$$\hat{q}_{1-\alpha}(x) := \inf\!\left\{v \in \mathbb{R} \cup \{+\infty\} : \sum_{i=1}^n p_i(x)\,\mathbb{I}(\tilde{V}_i \le v) + p_\infty(x)\,\mathbb{I}(+\infty \le v) \ge 1 - \alpha\right\}.$$

### Edge-Case Invariants

1. **Unattainable Coverage ($\hat{q}_{1-\alpha}(x) = +\infty$):** Whenever $\sum_{i \in \mathcal{I}_{c_0}} p_i(x) < 1 - \alpha$ (equivalently $p_\infty(x) > \alpha$, including when $\mathcal{I}_{c_0} = \emptyset$), $\hat{q}_{1-\alpha}(x) = +\infty$ and $\hat{L}(x) = -\infty$.
2. **Tie-Breaking Invariance:** Equal active score atoms share identical jump discontinuities in the discrete CDF, so $\hat{q}_{1-\alpha}(x)$ is invariant to permutation among tied scores.
3. **Kish Effective Sample Size Guard:** Calibration weight concentration is summarized by
   $$\hat{n}_{\mathrm{eff}}(w) := \begin{cases} \dfrac{(\sum_{i=1}^n w_i)^2}{\sum_{i=1}^n w_i^2}, & \sum_{i=1}^n w_i > 0, \\[8pt] 0, & \sum_{i=1}^n w_i = 0. \end{cases}$$
   When $\hat{n}_{\mathrm{eff}}(w) < \lceil 1 / \alpha \rceil - 1$, calibration emits a warning and marks the state underfilled, returning the conservative unbounded interval $(-\infty, +\infty)$.
