from rembg import new_session, remove


class U2NetModel:
    """Thin wrapper around a rembg session."""

    def __init__(self, session: object) -> None:
        self._session = session


def load_u2net_model() -> U2NetModel:
    """Load the U2Net model via rembg (downloads on first use, then caches)."""
    session = new_session("u2net")
    return U2NetModel(session)


def remove_background(image_data: bytes, model: U2NetModel) -> bytes:
    """Remove background from an image, returning a PNG with transparent background."""
    return remove(image_data, session=model._session)  # type: ignore[arg-type]
