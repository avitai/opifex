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


def _register_graph_pytrees() -> None:  # noqa: PLR0915
    """Register graph-related pytrees."""
    # Import here to avoid circular imports
    from opifex.geometry.topology.graphs import (
        GraphMessagePassing,
        GraphNeuralOperator,
        GraphTopology,
    )

    # Register GraphMessagePassing
    def graph_message_passing_tree_flatten(
        gmp: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten GraphMessagePassing for pytree operations."""
        children = (
            gmp.msg_w1,
            gmp.msg_b1,
            gmp.msg_w2,
            gmp.msg_b2,
            gmp.upd_w1,
            gmp.upd_b1,
            gmp.upd_w2,
            gmp.upd_b2,
        )
        aux_data = (gmp.node_dim, gmp.edge_dim, gmp.hidden_dim, gmp.output_dim)
        return children, aux_data

    def graph_message_passing_tree_unflatten(
        aux_data: tuple[Any, ...], children: tuple[Any, ...]
    ) -> Any:
        """Unflatten GraphMessagePassing from pytree operations."""
        node_dim, edge_dim, hidden_dim, output_dim = aux_data
        msg_w1, msg_b1, msg_w2, msg_b2, upd_w1, upd_b1, upd_w2, upd_b2 = children

        # Create new instance without calling __init__
        gmp = GraphMessagePassing.__new__(GraphMessagePassing)
        gmp.node_dim = node_dim
        gmp.edge_dim = edge_dim
        gmp.hidden_dim = hidden_dim
        gmp.output_dim = output_dim
        gmp.msg_w1 = msg_w1
        gmp.msg_b1 = msg_b1
        gmp.msg_w2 = msg_w2
        gmp.msg_b2 = msg_b2
        gmp.upd_w1 = upd_w1
        gmp.upd_b1 = upd_b1
        gmp.upd_w2 = upd_w2
        gmp.upd_b2 = upd_b2
        return gmp

    jax.tree_util.register_pytree_node(
        GraphMessagePassing,
        graph_message_passing_tree_flatten,
        graph_message_passing_tree_unflatten,
    )

    # Register GraphNeuralOperator
    def graph_neural_operator_tree_flatten(
        gno: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten GraphNeuralOperator for pytree operations."""
        # Flatten all mp_layers
        mp_layers_children = []
        for layer in gno.mp_layers:
            layer_children, _ = graph_message_passing_tree_flatten(layer)
            mp_layers_children.extend(layer_children)

        children = (
            gno.input_w,
            gno.input_b,
            gno.output_w,
            gno.output_b,
            *mp_layers_children,
        )
        aux_data = (gno.node_dim, gno.hidden_dim, gno.output_dim, gno.num_layers)
        return children, aux_data

    def graph_neural_operator_tree_unflatten(
        aux_data: tuple[Any, ...], children: tuple[Any, ...]
    ) -> Any:
        """Unflatten GraphNeuralOperator from pytree operations."""
        node_dim, hidden_dim, output_dim, num_layers = aux_data

        # Extract main parameters
        input_w, input_b, output_w, output_b = children[:4]

        # Extract mp_layers parameters (8 parameters per layer)
        mp_params = children[4:]
        mp_layers = []
        for i in range(num_layers):
            layer_params = mp_params[i * 8 : (i + 1) * 8]
            layer_aux = (
                hidden_dim,
                0,
                hidden_dim,
                hidden_dim,
            )  # edge_dim will be set properly
            layer = graph_message_passing_tree_unflatten(layer_aux, layer_params)
            mp_layers.append(layer)

        # Create new instance without calling __init__
        gno = GraphNeuralOperator.__new__(GraphNeuralOperator)
        gno.node_dim = node_dim
        gno.hidden_dim = hidden_dim
        gno.output_dim = output_dim
        gno.num_layers = num_layers
        gno.input_w = input_w
        gno.input_b = input_b
        gno.output_w = output_w
        gno.output_b = output_b
        gno.mp_layers = mp_layers
        return gno

    jax.tree_util.register_pytree_node(
        GraphNeuralOperator,
        graph_neural_operator_tree_flatten,
        graph_neural_operator_tree_unflatten,
    )

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
