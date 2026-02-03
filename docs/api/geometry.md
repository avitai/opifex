# Geometry API Reference

The `opifex.geometry` package provides tools for defining computational domains using Constructive Solid Geometry (CSG).

## Shapes

### Base Protocol

::: opifex.geometry.csg.Shape2D
    options:
        show_root_heading: true
        show_source: true

### Basic Shapes

::: opifex.geometry.csg.Rectangle
    options:
        show_root_heading: true
        show_source: true

::: opifex.geometry.csg.Circle
    options:
        show_root_heading: true
        show_source: true

::: opifex.geometry.csg.Polygon
    options:
        show_root_heading: true
        show_source: true

## CSG Operations

### Classes

::: opifex.geometry.csg.CSGUnion
    options:
        show_root_heading: true
        show_source: true

::: opifex.geometry.csg.CSGIntersection
    options:
        show_root_heading: true
        show_source: true

::: opifex.geometry.csg.CSGDifference
    options:
        show_root_heading: true
        show_source: true

### Functional API

::: opifex.geometry.csg.union
::: opifex.geometry.csg.intersection
::: opifex.geometry.csg.difference

## Boundary Analysis

::: opifex.geometry.csg.compute_boundary_normals
::: opifex.geometry.csg.sample_boundary_points

## Molecular Geometry

::: opifex.geometry.csg.MolecularGeometry
    options:
        show_root_heading: true
        show_source: true
