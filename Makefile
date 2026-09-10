.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE := docker compose
PSQL_USER ?= varuna
PSQL_DB   ?= varuna

.PHONY: help up down logs ps reset-db seed test demo fixtures psql lint frontend

help:
	@echo "VARUNA"
	@echo "  make up         docker compose up --build (db, redis, backend)"
	@echo "  make down       stop the stack, keep the volume"
	@echo "  make reset-db   DESTROY the db volume and re-apply schema.sql"
	@echo "  make seed       schema + scenes + AIS + demo scenarios"
	@echo "  make test       pytest + tsc --noEmit"
	@echo "  make demo       run all 3 scenarios headless, assert verdicts"
	@echo "  make fixtures   regenerate data/fixtures - HUMAN-INVOKED ONLY"
	@echo "  make frontend   npm run dev on the host (not containerised)"
	@echo "  make psql       psql shell into the database"

up:
	$(COMPOSE) up --build -d
	@echo "backend  http://localhost:8000/api/v1/health"
	@echo "frontend runs on the host: cd frontend && npm run dev"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# schema.sql is executed by the postgis container from
# /docker-entrypoint-initdb.d, which runs ONLY on an empty volume. Any edit to
# backend/app/db/schema.sql therefore requires this target - without it the
# change silently does not appear in a running database.
reset-db:
	@echo "Destroying the database volume and re-applying schema.sql..."
	$(COMPOSE) down -v
	$(COMPOSE) up -d db
	@echo "Done. Run 'make up' to bring the rest of the stack back."

seed:
	$(COMPOSE) exec -T backend python scripts/seed_db.py

test:
	$(COMPOSE) exec -T backend python -m pytest -q
	@if [ -d frontend/node_modules ]; then \
		cd frontend && npx tsc --noEmit; \
	else \
		echo "[test] frontend/node_modules absent - skipping tsc (run: cd frontend && npm install)"; \
	fi

demo:
	@echo "[demo] phase 1 stub - the release gate lands with the demo controller"
	@exit 0

fixtures:
	@echo "HUMAN-INVOKED ONLY. Fixtures are FROZEN once green (§9)."
	@echo "This regenerates data/fixtures from real engine output (§15)."
	$(COMPOSE) exec -T backend python scripts/build_fixtures.py

psql:
	$(COMPOSE) exec db psql -U $(PSQL_USER) -d $(PSQL_DB)

lint:
	$(COMPOSE) exec -T backend python -m ruff check app scripts ml

frontend:
	cd frontend && npm run dev
