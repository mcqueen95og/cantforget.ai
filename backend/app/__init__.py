"""cantforget.ai backend package.

``app`` is exposed lazily so that the face/identity modules can be imported
(and unit-tested) without FastAPI installed. ``uvicorn backend.app:app``
still works because uvicorn resolves the attribute on the module.
"""

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from .main import app

        return app
    raise AttributeError(name)
