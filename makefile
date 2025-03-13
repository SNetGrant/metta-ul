# Makefile for running pytest inside Docker container

# Docker image and container names
IMAGE_NAME = metta_ul
CONTAINER_NAME = metta_ul_run
CWD = $(shell pwd)

# Build the Docker image
build:
	@echo "Building Docker image: $(IMAGE_NAME)..."
	docker build . -t $(IMAGE_NAME)

# Stop and remove the existing container, then run pytest inside the container
test: 
	@echo "Stopping and removing existing container: $(CONTAINER_NAME)..."
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	@echo "Running pytest in the container..."
	docker run -it --mount type=bind,src=$(CWD),dst=/app --name $(CONTAINER_NAME)  $(IMAGE_NAME)  pytest -sx
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Default target
all: build test
