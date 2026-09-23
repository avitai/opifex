# Fields API Reference

JAX-native field abstractions for scientific computing on structured grids.

Two layouts live here, and which one a problem wants is a property of the problem rather
than a preference.

A **collocated** grid (`CenteredGrid`) stores every component at the cell centre. It is the
natural home for a scalar field and for anything sampled or interpolated pointwise. Its
limitation is structural: `gradient` and `divergence` are both two-point central
differences, so their composition reaches `i ± 2`, annihilates the three checkerboard modes
along with the constant, and cannot be made skew-adjoint at a zero-gradient boundary — the
`edge` padding gives the difference operator a diagonal entry, which no skew operator has.

A **staggered** grid (`StaggeredGrid`, the Arakawa-C or MAC arrangement) puts each velocity
component on the faces normal to its own axis. A boundary-normal face is then a boundary
condition rather than an unknown, so the gradient never needs a ghost pressure. Measured as
matrices, `divergence` is exactly minus the adjoint of `gradient` in the plain Euclidean
inner product under every boundary, the pressure operator is symmetric and negative
semi-definite, and its null space is the constants alone. That is what an incompressible
projection needs, so the flow solvers use it.

The two carry operators of the same names — `divergence`, `gradient` — for the two layouts,
and they are deliberately not merged into one namespace: they act on different objects, and
a flat import would let one silently shadow the other. Reach them by module.

## Collocated fields

::: opifex.fields

## Staggered fields

::: opifex.fields.staggered

## Incompressible projection on a staggered grid

::: opifex.fields.staggered_pressure

## Convection on a staggered grid

::: opifex.fields.staggered_convection

## Incompressible Navier-Stokes in time

::: opifex.fields.staggered_navier_stokes
