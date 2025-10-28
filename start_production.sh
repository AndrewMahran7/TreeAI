#!/bin/bash
# ArborNote Production Startup Script
# This script prepares and starts the application in production mode

set -e

echo "🚀 Starting ArborNote in Production Mode"
echo "========================================"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Do not run this script as root for security reasons"
    exit 1
fi

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if required environment files exist
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f ".env.production" ]; then
        cp .env.production .env
        echo "✅ Copied .env.production to .env"
        echo "📝 Please edit .env file with your actual values before proceeding"
        exit 1
    else
        echo "❌ No environment template found. Please create .env file."
        exit 1
    fi
fi

# Validate critical environment variables
echo "🔍 Validating environment configuration..."

# Source the environment file
set -a
source .env
set +a

# Check critical variables
MISSING_VARS=""

if [ -z "$FLASK_SECRET_KEY" ] || [ "$FLASK_SECRET_KEY" = "your-super-secret-key-change-this-in-production" ]; then
    MISSING_VARS="$MISSING_VARS FLASK_SECRET_KEY"
fi

if [ -z "$NEARMAP_API_KEY" ] || [ "$NEARMAP_API_KEY" = "your-nearmap-api-key-here" ]; then
    MISSING_VARS="$MISSING_VARS NEARMAP_API_KEY"
fi

if [ -n "$MISSING_VARS" ]; then
    echo "❌ Missing or invalid environment variables:$MISSING_VARS"
    echo "📝 Please update your .env file with actual values"
    exit 1
fi

echo "✅ Environment configuration validated"

# Create necessary directories
echo "📁 Creating application directories..."
mkdir -p logs outputs temp_files checkpoints static/uploads
chmod 755 logs outputs temp_files checkpoints static

# Set up log rotation
echo "🔄 Setting up log rotation..."
if command -v logrotate &> /dev/null; then
    cat > arbornote-logrotate << EOF
/$(pwd)/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(whoami)
    postrotate
        docker-compose exec arbornote kill -USR1 1 2>/dev/null || true
    endscript
}
EOF
    echo "✅ Log rotation configured"
else
    echo "⚠️  logrotate not found - logs will not be automatically rotated"
fi

# Check for model files
echo "🤖 Checking for AI model files..."
MODEL_COUNT=$(find checkpoints -name "*.ckpt" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "⚠️  No model checkpoint files found in checkpoints/ directory"
    echo "📝 The application will run in simulation mode without real AI detection"
else
    echo "✅ Found $MODEL_COUNT model checkpoint files"
fi

# Security check - ensure sensitive files are not world-readable
echo "🔒 Applying security permissions..."
chmod 600 .env 2>/dev/null || true
chmod 600 .env.* 2>/dev/null || true
find . -name "*.key" -exec chmod 600 {} \; 2>/dev/null || true
find . -name "*.pem" -exec chmod 600 {} \; 2>/dev/null || true

# Clean up old containers and images
echo "🧹 Cleaning up old Docker resources..."
docker-compose down --remove-orphans 2>/dev/null || true
docker system prune -f --volumes

# Build the application
echo "🔨 Building ArborNote application..."
if ! docker-compose build --no-cache; then
    echo "❌ Failed to build the application"
    exit 1
fi

# Start the application
echo "🚀 Starting ArborNote application..."
if ! docker-compose up -d; then
    echo "❌ Failed to start the application"
    exit 1
fi

# Wait for application to be ready
echo "⏳ Waiting for application to start..."
HEALTH_URL="http://localhost:4000/health"
MAX_WAIT=120  # 2 minutes
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -f $HEALTH_URL > /dev/null 2>&1; then
        echo "✅ Application is healthy and ready!"
        break
    fi
    
    echo "⏳ Waiting for application... ($WAIT_TIME/$MAX_WAIT seconds)"
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "❌ Application failed to start within $MAX_WAIT seconds"
    echo "📋 Checking logs..."
    docker-compose logs --tail=50
    exit 1
fi

# Display status and access information
echo ""
echo "🎉 ArborNote is now running in production mode!"
echo "========================================"
echo ""
echo "📊 Application Status:"
docker-compose ps

echo ""
echo "🌐 Access Information:"
echo "   Local URL: http://localhost:4000"
echo "   Health Check: http://localhost:4000/health"

if [ -n "$ALLOWED_HOSTS" ] && [ "$ALLOWED_HOSTS" != "your-domain.com,www.your-domain.com" ]; then
    echo "   Production URL: https://$(echo $ALLOWED_HOSTS | cut -d',' -f1)"
fi

echo ""
echo "📋 Useful Commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop app: docker-compose down"
echo "   Restart: docker-compose restart"
echo "   Update: git pull && docker-compose build --no-cache && docker-compose up -d"

echo ""
echo "🔍 Monitoring:"
echo "   Application logs: tail -f logs/arbornote.log"
echo "   System resources: docker stats"
echo "   Health status: curl http://localhost:4000/health"

echo ""
echo "✅ ArborNote production startup completed successfully!"

# Show final health check
echo ""
echo "🏥 Final Health Check:"
curl -s http://localhost:4000/health | python3 -m json.tool 2>/dev/null || echo "Health check endpoint not responding properly"

echo ""
echo "🎯 Your ArborNote application is ready for tree detection!"
echo "📝 Remember to monitor your logs and system resources regularly."