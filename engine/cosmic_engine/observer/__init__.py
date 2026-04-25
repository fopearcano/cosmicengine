"""Observer-dependent reality system.

Phase 33 contribution: multiple observers per universe. Each observer
carries its own pose, motion, perception knobs, and optional spacetime
or AI warp models — so the same physical state produces a different
:class:`RealityView` per observer. The single-observer pipelines in
:mod:`cosmic_engine.runtime` keep working unchanged: an empty
:class:`ObserverManager` is the default, and the new ``render_for_observer``
path is opt-in.
"""

from cosmic_engine.observer.observer import Observer
from cosmic_engine.observer.observer_manager import ObserverManager
from cosmic_engine.observer.reality_view import RealityView

__all__ = [
    "Observer",
    "ObserverManager",
    "RealityView",
]
