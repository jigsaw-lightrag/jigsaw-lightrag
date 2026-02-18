from fastapi import FastAPI, Depends
import signal
from uvicorn import run
import sys
from contextlib import asynccontextmanager 


from src.app.router import (
    jigsaw_api
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup_event()
    yield
    await shutdown_event()

app = FastAPI(
    lifespan=lifespan
)  

# relative or absolute path
app.include_router(jigsaw_api.router)

async def startup_event():
    print("lifespan")
    pass

async def shutdown_event():
    print("Performing clean shutdown...")
    print("Closing database connection...")
    print("Releasing resources...")


def handle_shutdown(signum, frame):
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_shutdown)

    run("app:app", host="0.0.0.0", port=8000)
