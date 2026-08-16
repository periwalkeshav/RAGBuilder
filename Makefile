.DEFAULT_GOAL := help
COMPOSE := docker compose
API := $(COMPOSE) exec -T api python -m ragbuilder.cli
PSQL := $(COMPOSE) exec -T postgres psql -U ragbuilder -d ragbuilder

.PHONY: help up down clean build logs ps health test test-unit test-integration lint \
        download ingest embed setup ask search evaluate stats psql reset-vectors ui-local

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build the API and UI images
	$(COMPOSE) build

up: ## Start the stack (Ollama must already be running on the host)
	@command -v ollama >/dev/null || echo "WARNING: ollama not found on PATH - the API will be degraded"
	$(COMPOSE) up -d --build
	@echo
	@echo "  API docs   http://localhost:8000/docs"
	@echo "  Chat UI    http://localhost:8501"
	@echo "  Qdrant     http://localhost:6333/dashboard"
	@echo "  MLflow     http://localhost:5001"

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

clean: ## Stop and delete all data, including the model cache
	$(COMPOSE) down -v --remove-orphans

ps: ## Service status
	$(COMPOSE) ps

logs: ## Tail the API logs
	$(COMPOSE) logs -f api

health: ## Component health
	@curl -s http://localhost:8000/health | python3 -m json.tool

setup: ## One command from empty to answering questions
	$(MAKE) up
	@sleep 15
	$(API) ingest --strategies sentence fixed
	$(API) embed --strategy sentence --recreate
	$(API) embed --strategy fixed
	$(MAKE) stats

download: ## Download the corpus PDFs
	$(API) download

ingest: ## Parse and chunk (sentence + fixed)
	$(API) ingest --strategies sentence fixed

ingest-all: ## Parse and chunk all three strategies (semantic is slow)
	$(API) ingest --all-strategies

embed: ## Embed the sentence chunks
	$(API) embed --strategy sentence

embed-all: ## Embed every strategy
	$(API) embed --all-strategies

reset-vectors: ## Drop and rebuild the Qdrant collection
	$(API) embed --strategy sentence --recreate --force

ask: ## make ask Q="Wie viele Urlaubstage stehen mir zu?"
	@$(API) ask "$(Q)"

search: ## make search Q="Kündigungsfrist"
	@$(API) search "$(Q)"

evaluate: ## Score all strategies and log to MLflow
	$(API) evaluate --strategies fixed sentence semantic --output data/evaluation.json

evaluate-fast: ## Retrieval-only evaluation (no LLM judging)
	$(API) evaluate --strategies fixed sentence --no-score

stats: ## Knowledge base statistics
	@$(API) stats

psql: ## SQL shell
	$(COMPOSE) exec postgres psql -U ragbuilder -d ragbuilder

test: test-unit ## Alias for test-unit

test-unit: ## Unit tests (no stack required)
	docker run --rm ragbuilder-api python -m pytest tests/ -q --ignore=tests/test_integration.py

test-integration: ## Integration tests against the running stack
	docker run --rm --network ragbuilder -e RUN_INTEGRATION=1 -e API_URL=http://api:8000 \
		ragbuilder-api python -m pytest tests/test_integration.py -v

lint: ## flake8 + black --check
	docker run --rm -v "$(PWD)":/w -w /w python:3.11-slim sh -c \
		"pip install -q flake8==7.1.0 black==24.4.2 && \
		 flake8 ragbuilder ui tests && black --check --line-length 110 ragbuilder ui tests"

ui-local: ## Run the Streamlit UI against a local API
	API_URL=http://localhost:8000 streamlit run ui/app.py
