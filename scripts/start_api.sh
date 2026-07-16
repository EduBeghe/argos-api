#!/bin/sh
set -eu

python -m alembic upgrade head
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
