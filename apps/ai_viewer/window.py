"""Lightweight tkinter-based viewer window.

Tkinter is in the Python standard library, so no extra dependency is
needed beyond Pillow (which we use for the actual image conversion).
Designed to degrade gracefully on headless environments: if a display
is not available the window flips to ``available = False`` and every
later ``show_frame`` / ``update`` is a no-op.
"""

from __future__ import annotations

from typing import Any


class ViewerWindow:
    """A single-image window. Call :meth:`show_frame` then :meth:`update`."""

    def __init__(
        self,
        width: int,
        height: int,
        title: str = "CosmicEngine AI Viewer",
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("ViewerWindow width and height must be positive")
        self.width = width
        self.height = height
        self.title = title
        self.available: bool = True
        self._closed: bool = False
        self._root: Any = None
        self._label: Any = None
        self._photo: Any = None

    def _ensure_open(self) -> None:
        if self._root is not None or not self.available or self._closed:
            return
        try:
            import tkinter as tk
        except Exception:
            self.available = False
            return
        try:
            root = tk.Tk()
            root.title(self.title)
            root.geometry(f"{self.width}x{self.height}")
            root.protocol("WM_DELETE_WINDOW", self._on_close)
            label = tk.Label(root)
            label.pack()
        except Exception:
            self.available = False
            self._root = None
            return
        self._root = root
        self._label = label

    def show_frame(self, image: Any) -> None:
        """Display ``image`` (a PIL :class:`Image.Image`)."""
        if not self.available or self._closed:
            return
        self._ensure_open()
        if self._root is None:
            return
        try:
            from PIL import ImageTk
        except Exception:
            self.available = False
            return
        try:
            self._photo = ImageTk.PhotoImage(image)
            self._label.configure(image=self._photo)
        except Exception:
            self.available = False

    def update(self) -> None:
        """Pump tkinter events without blocking."""
        if not self.available or self._closed or self._root is None:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            self.available = False

    def close(self) -> None:
        """Tear down the window. Safe to call multiple times."""
        self._closed = True
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
            self._label = None
            self._photo = None

    def _on_close(self) -> None:
        self._closed = True
