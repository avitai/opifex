"""PyTree registration utilities for geometric objects."""

import logging
from typing import Any

import jax


_logger = logging.getLogger(__name__)


def _register_manifold_pytrees(RiemannianManifold) -> None:
    """Register manifold-related pytrees."""

    def riemannian_tree_flatten(
        manifold: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten RiemannianManifold for pytree operations."""
        children = ()
        aux_data = (
            manifold.dimension,
            manifold.embedding_dimension,
            manifold.metric_function,
            manifold.coordinate_chart,
            manifold.inverse_chart,
        )
        return children, aux_data

    def riemannian_tree_unflatten(aux_data: tuple[Any, ...], _: tuple[Any, ...]) -> Any:
        """Unflatten RiemannianManifold from pytree operations."""
        (
            dimension,
            embedding_dimension,
            metric_function,
            coordinate_chart,
            inverse_chart,
        ) = aux_data
        return RiemannianManifold(
            dimension=dimension,
            metric_function=metric_function,
            embedding_dimension=embedding_dimension,
            coordinate_chart=coordinate_chart,
            inverse_chart=inverse_chart,
        )

    jax.tree_util.register_pytree_node(
        RiemannianManifold, riemannian_tree_flatten, riemannian_tree_unflatten
    )


def _register_shape_pytrees(Rectangle, Circle, Polygon) -> None:
    """Register basic shape pytrees."""

    # Register Rectangle
    def rectangle_tree_flatten(
        rect: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten Rectangle for pytree operations."""
        children = (rect.center,)
        aux_data = (rect.width, rect.height)
        return children, aux_data

    def rectangle_tree_unflatten(aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        """Unflatten Rectangle from pytree operations."""
        (center,) = children
        width, height = aux_data
        rect = Rectangle.__new__(Rectangle)
        rect.center = center
        rect.width = width
        rect.height = height
        rect.x_min = rect.center[0] - rect.width / 2
        rect.x_max = rect.center[0] + rect.width / 2
        rect.y_min = rect.center[1] - rect.height / 2
        rect.y_max = rect.center[1] + rect.height / 2
        return rect

    jax.tree_util.register_pytree_node(Rectangle, rectangle_tree_flatten, rectangle_tree_unflatten)

    # Register Circle
    def circle_tree_flatten(circle: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten Circle for pytree operations."""
        children = (circle.center,)
        aux_data = (circle.radius,)
        return children, aux_data

    def circle_tree_unflatten(aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        """Unflatten Circle from pytree operations."""
        (center,) = children
        (radius,) = aux_data
        circle = Circle.__new__(Circle)
        circle.center = center
        circle.radius = radius
        return circle

    jax.tree_util.register_pytree_node(Circle, circle_tree_flatten, circle_tree_unflatten)

    # Register Polygon
    def polygon_tree_flatten(
        polygon: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten Polygon for pytree operations."""
        children = (polygon.vertices,)
        aux_data = ()  # Empty tuple for no auxiliary data
        return children, aux_data

    def polygon_tree_unflatten(_: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        """Unflatten Polygon from pytree operations."""
        (vertices,) = children
        polygon = Polygon.__new__(Polygon)
        polygon.vertices = vertices
        return polygon

    jax.tree_util.register_pytree_node(Polygon, polygon_tree_flatten, polygon_tree_unflatten)


def _register_csg_pytrees(CSGUnion, CSGIntersection, CSGDifference) -> None:
    """Register CSG operation pytrees."""

    # Register CSGUnion
    def csg_union_tree_flatten(
        union: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten CSGUnion for pytree operations."""
        children = (union.shape_a, union.shape_b)
        aux_data = ()  # Empty tuple for no auxiliary data
        return children, aux_data

    def csg_union_tree_unflatten(_: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        """Unflatten CSGUnion from pytree operations."""
        shape_a, shape_b = children
        return CSGUnion(shape_a, shape_b)

    jax.tree_util.register_pytree_node(CSGUnion, csg_union_tree_flatten, csg_union_tree_unflatten)

    # Register CSGIntersection
    def csg_intersection_tree_flatten(
        intersection: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten CSGIntersection for pytree operations."""
        children = (intersection.shape_a, intersection.shape_b)
        aux_data = ()  # Empty tuple for no auxiliary data
        return children, aux_data

    def csg_intersection_tree_unflatten(_: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        """Unflatten CSGIntersection from pytree operations."""
        shape_a, shape_b = children
        return CSGIntersection(shape_a, shape_b)

    jax.tree_util.register_pytree_node(
        CSGIntersection,
        csg_intersection_tree_flatten,
        csg_intersection_tree_unflatten,
    )

    # Register CSGDifference
    def csg_difference_tree_flatten(
        difference: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten CSGDifference for pytree operations."""
        children = (difference.shape_a, difference.shape_b)
        aux_data = ()  # Empty tuple for no auxiliary data
        return children, aux_data

    def csg_difference_tree_unflatten(_: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        """Unflatten CSGDifference from pytree operations."""
        shape_a, shape_b = children
        return CSGDifference(shape_a, shape_b)

    jax.tree_util.register_pytree_node(
        CSGDifference, csg_difference_tree_flatten, csg_difference_tree_unflatten
    )


def _register_periodic_pytrees(PeriodicCell) -> None:
    """Register periodic cell pytrees."""

    def periodic_cell_tree_flatten(
        cell: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten PeriodicCell for pytree operations."""
        children = (cell.lattice_vectors,)
        aux_data = ()  # Empty tuple for no auxiliary data
        return children, aux_data

    def periodic_cell_tree_unflatten(_: tuple[Any, ...], children: tuple[Any, ...]) -> Any:
        """Unflatten PeriodicCell from pytree operations."""
        (lattice_vectors,) = children
        return PeriodicCell(lattice_vectors)

    jax.tree_util.register_pytree_node(
        PeriodicCell, periodic_cell_tree_flatten, periodic_cell_tree_unflatten
    )


def _register_molecular_pytrees(MolecularGeometry) -> None:
    """Register molecular geometry pytrees."""

    def molecular_geometry_tree_flatten(
        mol_geom: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten MolecularGeometry for pytree operations."""
        children = (mol_geom.positions,)
        aux_data = (mol_geom.atomic_numbers,)
        return children, aux_data

    def molecular_geometry_tree_unflatten(
        aux_data: tuple[Any, ...], children: tuple[Any, ...]
    ) -> Any:
        """Unflatten MolecularGeometry from pytree operations."""
        (positions,) = children
        (atomic_numbers,) = aux_data
        mol_geom = MolecularGeometry.__new__(MolecularGeometry)
        mol_geom.positions = positions
        mol_geom.atomic_numbers = atomic_numbers
        mol_geom.num_atoms = len(mol_geom.atomic_numbers)
        return mol_geom

    jax.tree_util.register_pytree_node(
        MolecularGeometry,
        molecular_geometry_tree_flatten,
        molecular_geometry_tree_unflatten,
    )


def _register_graph_pytrees() -> None:
    """Register graph-related pytrees."""
    # Import here to avoid circular imports
    from opifex.geometry.topology.graphs import GraphTopology

    # Register GraphTopology.
    # edge_features is a traced child (not static aux_data) so it survives a
    # flatten/unflatten round-trip and stays a differentiable leaf under
    # jit/grad/vmap.  A None edge_features remains None because JAX treats it as
    # an empty subtree.
    def graph_topology_tree_flatten(gt: Any) -> tuple[tuple[Any, ...], None]:
        """Flatten GraphTopology for pytree operations."""
        children = (gt.nodes, gt.edges, gt.edge_features, gt.adjacency_matrix)
        return children, None

    def graph_topology_tree_unflatten(aux_data: None, children: tuple[Any, ...]) -> Any:
        """Unflatten GraphTopology from pytree operations."""
        del aux_data
        nodes, edges, edge_features, adjacency_matrix = children

        # Create new instance without calling __init__ to avoid recomputing adjacency.
        gt = GraphTopology.__new__(GraphTopology)
        gt.nodes = nodes
        gt.edges = edges
        gt.edge_features = edge_features
        gt.adjacency_matrix = adjacency_matrix
        return gt

    jax.tree_util.register_pytree_node(
        GraphTopology, graph_topology_tree_flatten, graph_topology_tree_unflatten
    )


# Register all pytrees when module is imported
def register_geometric_pytrees() -> None:
    """Register all geometric objects as JAX pytrees."""
    # Import all geometric classes
    from opifex.geometry.csg import (
        Circle,
        CSGDifference,
        CSGIntersection,
        CSGUnion,
        MolecularGeometry,
        PeriodicCell,
        Polygon,
        Rectangle,
    )
    from opifex.geometry.manifolds.riemannian import RiemannianManifold

    # Register manifold pytrees
    _register_manifold_pytrees(RiemannianManifold)

    # Register shape pytrees
    _register_shape_pytrees(Rectangle, Circle, Polygon)

    # Register CSG pytrees
    _register_csg_pytrees(CSGUnion, CSGIntersection, CSGDifference)

    # Register periodic pytrees
    _register_periodic_pytrees(PeriodicCell)

    # Register molecular pytrees
    _register_molecular_pytrees(MolecularGeometry)

    # Register graph pytrees.  Only a genuinely missing graph module is tolerated
    # (graph neural networks then stay disabled); any other failure must surface.
    try:
        _register_graph_pytrees()
    except ImportError:
        _logger.warning(
            "Graph pytree registration skipped: graph module unavailable.", exc_info=True
        )


# Call registration when module is imported
register_geometric_pytrees()
