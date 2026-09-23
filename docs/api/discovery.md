# Discovery API Reference

Equation discovery framework for recovering governing equations from data.

## SINDy Module

::: opifex.discovery.sindy

## Bayesian SINDy

A regularised-horseshoe posterior over the SINDy candidate-library coefficients (Hirsh,
Barajas-Solano & Kutz 2021, arXiv:2107.02107; Piironen & Vehtari 2017), sampled with
BlackJAX NUTS. The error model is on the derivatives, so `sigma` is a derivative-noise
scale; the integrated model of Hirsh et al. eq. 2.4 is not implemented. Import it from
`opifex.discovery.sindy.bayesian`: the SINDy package does not re-export it, so importing
SINDy does not load BlackJAX.

::: opifex.discovery.sindy.bayesian

## Symbolic Discovery

::: opifex.discovery.symbolic
