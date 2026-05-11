import sys
import warnings

import uvicorn

from src import config
from src.server import Models, create_app
from src.services.face_detection_service import load_face_detection_model
from src.services.u2net_service import load_u2net_model

warnings.filterwarnings("ignore")


def main() -> None:
    if config.CORS_ORIGIN == "*":
        print(
            "WARNING: CORS_ORIGIN is not set — all origins are allowed. "
            "Set CORS_ORIGIN in .env before deploying to production.",
            file=sys.stderr,
        )

    print("Loading AI models...")
    u2net = load_u2net_model()
    face_detection = load_face_detection_model()
    print("Models loaded.")

    app = create_app(Models(u2net=u2net, face_detection=face_detection))
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
