"""ASGI entry for uvicorn.

Request handlers are synchronous Django views. Uvicorn runs several of them at
once through Django's thread pool. The board reader stays on its own thread.
"""

import os
import threading

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SAMS.settings")

from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
from django.core.asgi import get_asgi_application

django_application = ASGIStaticFilesHandler(get_asgi_application())


def _resume_recordings() -> None:
    """Finish live sessions and analyse uploads that a restart left half done."""
    from monitoring.services import live_recorder, video_analysis

    live_recorder.finalize_orphans()
    video_analysis.resume_pending()


async def _lifespan(receive, send) -> None:
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            from hardware.bringup import prepare_environment, start_hardware

            prepare_environment()
            start_hardware()
            threading.Thread(target=_resume_recordings, name="sams-resume", daemon=True).start()
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            from hardware.stream import get_stream

            try:
                get_stream().stop()
            except Exception:
                pass
            await send({"type": "lifespan.shutdown.complete"})
            return


async def application(scope, receive, send):
    if scope["type"] == "lifespan":
        await _lifespan(receive, send)
        return
    await django_application(scope, receive, send)
