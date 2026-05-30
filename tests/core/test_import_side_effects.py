"""Import-time side-effect guards (Rule 13).

Each target module must mutate **nothing global** when imported; the behaviour
stays available through an explicit call (a factory, a registration function,
or an opt-in setup function).

Side effects are verified in a *fresh* subprocess interpreter so that prior
imports during the pytest session cannot mask a regression. Each test snapshots
the relevant global state (``os.environ`` keys, a ``jax.config`` flag, the root
logger handlers, the singleton registry, module attributes) immediately after a
bare ``import`` and asserts the import did not touch it.

A companion test per module exercises the explicit-call path to prove the moved
behaviour is still reachable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_fresh(snippet: str) -> str:
    """Run ``snippet`` in a fresh interpreter and return its stripped stdout.

    A new process guarantees no module is pre-imported, so import-time side
    effects (or their absence) are observed cleanly. ``check=True`` surfaces a
    non-zero exit (e.g. an ``AssertionError``) as a test failure with the child
    stderr attached.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


# ---------------------------------------------------------------------------
# 1. ``opifex`` top-level package — auto-configure is opt-in (default OFF).
# ---------------------------------------------------------------------------


def test_importing_opifex_does_not_mutate_jax_config() -> None:
    """``import opifex`` must not flip ``jax_enable_x64`` or set the cache dir.

    Auto-configure is opt-in: with ``OPIFEX_AUTO_CONFIGURE`` unset, importing
    the package must leave ``jax.config`` untouched relative to a bare
    ``import jax``.
    """
    snippet = """
        import os
        os.environ.pop("OPIFEX_AUTO_CONFIGURE", None)
        import jax
        before_cache = jax.config.jax_compilation_cache_dir
        before_x64 = jax.config.jax_enable_x64
        import opifex
        after_cache = jax.config.jax_compilation_cache_dir
        after_x64 = jax.config.jax_enable_x64
        assert after_cache == before_cache, (before_cache, after_cache)
        assert after_x64 == before_x64, (before_x64, after_x64)
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


def test_importing_opifex_does_not_mutate_xla_flags_env() -> None:
    """``import opifex`` must not append to ``os.environ['XLA_FLAGS']``."""
    snippet = """
        import os
        os.environ.pop("OPIFEX_AUTO_CONFIGURE", None)
        before = os.environ.get("XLA_FLAGS")
        import opifex
        after = os.environ.get("XLA_FLAGS")
        assert after == before, (before, after)
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


def test_explicit_argument_value_one_does_not_auto_configure() -> None:
    """Auto-configure is opt-in: even ``OPIFEX_AUTO_CONFIGURE=1`` must NOT
    configure JAX at import time (only an explicit call does)."""
    snippet = """
        import os
        os.environ["OPIFEX_AUTO_CONFIGURE"] = "1"
        import jax
        before = jax.config.jax_compilation_cache_dir
        import opifex
        after = jax.config.jax_compilation_cache_dir
        assert after == before, (before, after)
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


def test_setup_jax_optimization_is_callable_and_configures() -> None:
    """The explicit ``setup_jax_optimization`` call still configures JAX."""
    snippet = """
        import jax
        import opifex
        opifex.setup_jax_optimization()
        assert jax.config.jax_compilation_cache_dir is not None
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


# ---------------------------------------------------------------------------
# 2. ``opifex.neural.bayesian`` — registration is explicit, not at import.
# ---------------------------------------------------------------------------


def test_importing_bayesian_does_not_populate_registry() -> None:
    """Importing the package must not register the model capabilities.

    The shared singleton ``UQRegistry`` must be empty of the bayesian model
    names right after import — registration moved to an explicit function.
    """
    snippet = """
        from opifex.uncertainty.registry import UQRegistry
        UQRegistry.reset()
        import opifex.neural.bayesian  # noqa: F401
        registry = UQRegistry()
        assert "model:ProbabilisticPINN" not in registry
        assert "model:MultiFidelityPINN" not in registry
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


def test_register_bayesian_capabilities_populates_registry() -> None:
    """The explicit registration function populates the supplied registry."""
    snippet = """
        from opifex.uncertainty.registry import UQRegistry
        from opifex.neural.bayesian import register_bayesian_capabilities
        UQRegistry.reset()
        registry = UQRegistry()
        register_bayesian_capabilities(registry)
        assert "model:ProbabilisticPINN" in registry
        assert "model:MultiFidelityPINN" in registry
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


def test_register_bayesian_capabilities_is_idempotent() -> None:
    """Calling the registration function twice must not raise on duplicates."""
    snippet = """
        from opifex.uncertainty.registry import UQRegistry
        from opifex.neural.bayesian import register_bayesian_capabilities
        UQRegistry.reset()
        registry = UQRegistry()
        register_bayesian_capabilities(registry)
        register_bayesian_capabilities(registry)  # must be a no-op, not ValueError
        assert "model:ProbabilisticPINN" in registry
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


# ---------------------------------------------------------------------------
# 3. ``opifex.deployment.server`` — no app / middleware / state at import.
# ---------------------------------------------------------------------------


def test_import_server_does_not_construct_app() -> None:
    """Importing the server module must not build a module-level ``app``."""
    snippet = """
        import opifex.deployment.server as server
        assert not hasattr(server, "app"), "module-level FastAPI app present"
        assert not hasattr(server, "app_state"), "module-level app_state present"
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


def test_server_module_has_no_module_level_basic_config() -> None:
    """``logging.basicConfig`` must live inside :func:`main`, not at module scope.

    A runtime root-logger snapshot is unreliable here: a transitive dependency
    (``orbax.checkpoint`` -> ``absl.logging``) installs a default root handler
    when ``core_serving`` is imported, which would mask opifex's own effect.
    The structural guard parses the source and asserts that every
    ``logging.basicConfig`` call is nested inside a function (``main``) — never
    executed merely by importing the module.
    """
    import ast
    import inspect

    from opifex.deployment import server

    tree = ast.parse(inspect.getsource(server))
    module_level_basic_config = [
        node
        for node in tree.body  # only top-level statements, not function bodies
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "basicConfig"
    ]
    assert not module_level_basic_config, "logging.basicConfig runs at server import time"


def test_create_app_builds_configured_fastapi() -> None:
    """The ``create_app`` factory returns a configured FastAPI instance."""
    snippet = """
        from opifex.deployment.server import create_app
        app = create_app()
        assert "Opifex" in app.title
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


# ---------------------------------------------------------------------------
# 4. ``tensorly_integration`` — does not pin JAX_PLATFORM_NAME at import.
# ---------------------------------------------------------------------------


def test_import_tensorly_integration_does_not_pin_platform() -> None:
    """Importing the module must not set ``JAX_PLATFORM_NAME`` in the env."""
    snippet = """
        import os
        os.environ.pop("JAX_PLATFORM_NAME", None)
        import opifex.neural.operators.fno.tensorly_integration  # noqa: F401
        assert "JAX_PLATFORM_NAME" not in os.environ, os.environ.get("JAX_PLATFORM_NAME")
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"


# ---------------------------------------------------------------------------
# 5. ``core.testing_infrastructure`` — no root-logger reconfiguration.
# ---------------------------------------------------------------------------


def test_import_testing_infrastructure_does_not_call_basic_config() -> None:
    """Importing the module must not add handlers to the root logger."""
    snippet = """
        import logging
        before = list(logging.getLogger().handlers)
        import opifex.core.testing_infrastructure  # noqa: F401
        after = list(logging.getLogger().handlers)
        assert after == before, (before, after)
        print("ok")
    """
    assert _run_fresh(snippet) == "ok"
