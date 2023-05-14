import uvicorn
import argparse
from config import LLM_PORT, UVICORN_RELOAD, UVICORN_LOGLEVEL, UVICORN_WORKERS

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=int(LLM_PORT))
parser.add_argument('--reload', action="store_true", default=UVICORN_RELOAD)
parser.add_argument('--log_level', type=str, default=UVICORN_LOGLEVEL)
parser.add_argument('--workers', type=int, default=UVICORN_WORKERS)
args = parser.parse_args()

if __name__ == "__main__":
    uvicorn.run(
        app="api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers,
    )
