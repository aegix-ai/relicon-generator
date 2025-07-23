#!/bin/bash

# Relicon AI Ad Creator - Startup Script
# Revolutionary AI-powered ad creation system

set -e

echo "🚀 Starting Relicon AI Ad Creator..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before continuing."
    echo "   Required: OPENAI_API_KEY, LUMA_API_KEY (optional: ELEVENLABS_API_KEY)"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_wait=60
    local wait_time=0
    
    echo "⏳ Waiting for $service to be ready..."
    
    while ! nc -z localhost $port >/dev/null 2>&1; do
        if [ $wait_time -ge $max_wait ]; then
            echo "❌ $service failed to start within $max_wait seconds"
            exit 1
        fi
        sleep 2
        wait_time=$((wait_time + 2))
    done
    
    echo "✅ $service is ready!"
}

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads outputs/temp outputs/audio outputs/video outputs/final

# Build and start services
echo "🏗️ Building and starting services..."
docker compose up -d --build

# Wait for services to be ready
wait_for_service "PostgreSQL" 5432
wait_for_service "Redis" 6379
wait_for_service "Backend API" 8000
wait_for_service "Frontend" 3000

echo ""
echo "🎉 Relicon AI Ad Creator is now running!"
echo ""
echo "🌐 Access Points:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo "   Health:    http://localhost:8000/health"
echo ""
echo "📊 Service Status:"
docker compose ps

echo ""
echo "📋 Quick Commands:"
echo "   View logs:     docker compose logs -f"
echo "   Stop services: docker compose down"
echo "   Restart:       docker compose restart"
echo "   Backend logs:  docker compose logs -f backend"
echo "   Frontend logs: docker compose logs -f frontend"
echo ""
echo "🔧 Development:"
echo "   Backend shell: docker compose exec backend bash"
echo "   Database:      docker compose exec postgres psql -U postgres -d relicon"
echo "   Redis CLI:     docker compose exec redis redis-cli"
echo ""
echo "✨ Create your first revolutionary ad at http://localhost:3000"
echo "" 