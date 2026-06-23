# Host-Inference-Models — deployment shortcuts.
#
# Every target applies the NVIDIA override (docker-compose.nvidia.yml) on top of
# the base compose, so GPU placement (e.g. audio-bark pinned to its configured
# GPU via NVIDIA_VISIBLE_DEVICES) is ALWAYS in effect. Running bare
# `docker compose up` skips the override and silently falls back to CPU — use
# these targets instead. Edit the device map in docker-compose.nvidia.yml.
#
# Examples:
#   make up                  # start everything on GPU
#   make build-audio-bark    # rebuild one service
#   make up-audio-bark       # (re)create one service with GPU placement
#   make logs-model-gateway  # follow one service's logs

COMPOSE := docker compose -f docker-compose.yml -f docker-compose.nvidia.yml

.PHONY: build up down restart ps logs help

help: ## List available targets
	@grep -E '^[a-zA-Z%-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | sort

build: ## Build all images (with nvidia override)
	$(COMPOSE) build

up: ## Start all services detached (GPU placement applied)
	$(COMPOSE) up -d

down: ## Stop and remove all services
	$(COMPOSE) down

restart: ## Restart all services
	$(COMPOSE) restart

ps: ## Show service status
	$(COMPOSE) ps

logs: ## Follow logs for all services
	$(COMPOSE) logs -f

build-%: ## Build one service, e.g. make build-audio-bark
	$(COMPOSE) build $*

up-%: ## (Re)create one service with GPU placement, e.g. make up-audio-bark
	$(COMPOSE) up -d $*

logs-%: ## Follow logs for one service, e.g. make logs-audio-bark
	$(COMPOSE) logs -f $*
