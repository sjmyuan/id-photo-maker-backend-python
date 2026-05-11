from PIL import Image
from PIL.Image import Image as PILImage
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


def remove_background(img: PILImage, model: U2NetModel) -> PILImage:
    """Remove background, accepting and returning a PIL Image to avoid redundant
    PNG encode/decode round-trips inside rembg.
    """
    result = remove(img, session=model._session)
    # rembg returns a PIL Image when given one; cast for type checker
    if not isinstance(result, Image.Image):
        raise TypeError(f"rembg returned unexpected type: {type(result)}")
    return result
