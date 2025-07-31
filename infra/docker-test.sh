#!/bin/bash
# Docker Test Script for Relicon AI Video Generator

echo "Testing Relicon Docker Setup..."
echo "=================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Check if required files exist
echo "Checking required files..."
required_files=("Dockerfile" "docker-compose.yml" "requirements.txt" ".env.example")
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "$file exists"
    else
        echo "$file missing"
        exit 1
    fi
done

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo ".env file not found. Creating from example..."
    cp .env.example .env
    echo "Please edit .env file with your API keys before running docker-compose up"
fi

# Build the Docker image
echo -e "\nBuilding Docker image..."
docker-compose build

if [[ $? -eq 0 ]]; then
    echo "Docker image built successfully!"
    
    echo -e "\nReady to deploy! Run these commands:"
    echo "   docker-compose up -d    # Start in background"
    echo "   docker-compose logs -f  # View logs"
    echo "   docker-compose down     # Stop system"
    
    echo -e "\n📱 Access points:"
    echo "   Application: http://localhost:8080"
    echo "   Health Check: http://localhost:8080/health"
    echo "   Videos: ./outputs/ directory"
    
else
    echo "Docker build failed. Check the error messages above."
    exit 1
fi

echo -e "\nDocker setup complete! System ready for production deployment."
