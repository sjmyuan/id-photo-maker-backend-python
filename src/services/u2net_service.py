from rembg import new_session, remove

from src import config


class U2NetModel:
    """Thin wrapper around a rembg session."""

    def __init__(self, session: object) -> None:
        self._session = session


def load_u2net_model() -> U2NetModel:
    """Load the background removal model via rembg
    (expects model pre-downloaded by download-models.sh).
    """
    session = new_session(config.BG_REMOVAL_MODEL)
    return U2NetModel(session)


def remove_background(image_data: bytes, model: U2NetModel) -> bytes:
    """Remove background from an image, returning a PNG with transparent background."""
    return remove(image_data, session=model._session)  # type: ignore[arg-type]
