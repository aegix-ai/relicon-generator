# Relicon AI Ad Creator - Production Deployment Guide

🚀 **Production deployment guide for the revolutionary AI ad creation system**

## Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │     CDN         │    │   Monitoring    │
│   (Nginx/HAP)   │    │  (CloudFlare)   │    │  (Grafana)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Swarm / K8s Cluster                 │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Frontend      │    Backend      │       Services              │
│   (React)       │   (FastAPI)     │                             │
│   Port: 3000    │   Port: 8000    │  ┌─────────────────────────┐│
│                 │                 │  │  PostgreSQL (Primary)   ││
│                 │   ┌───────────┐ │  │  Port: 5432             ││
│                 │   │  Celery   │ │  │                         ││
│                 │   │  Workers  │ │  │  PostgreSQL (Replica)  ││
│                 │   │           │ │  │  Port: 5433             ││
│                 │   └───────────┘ │  │                         ││
│                 │                 │  │  Redis Cluster          ││
│                 │                 │  │  Port: 6379-6384        ││
│                 │                 │  └─────────────────────────┘│
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Deployment Options

### Option 1: Docker Compose (Small-Medium Scale)

#### Server Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 100GB+ SSD
- **Network**: High-speed internet for AI APIs

#### Quick Deployment
```bash
# 1. Clone repository
git clone <your-repo-url>
cd relicon-rewrite

# 2. Set up environment
cp .env.example .env
# Edit .env with production values

# 3. Deploy
chmod +x start.sh
./start.sh

# 4. Configure reverse proxy (see Nginx section below)
```

### Option 2: Kubernetes (Large Scale)

#### Prerequisites
- Kubernetes cluster (GKE, EKS, AKS, or self-managed)
- kubectl configured
- Helm 3.x installed

#### Deployment Steps
```bash
# 1. Create namespace
kubectl create namespace relicon-ai

# 2. Create secrets
kubectl create secret generic relicon-secrets \
  --from-literal=DATABASE_URL="postgresql://user:pass@host:5432/relicon" \
  --from-literal=OPENAI_API_KEY="your-key" \
  --from-literal=LUMA_API_KEY="your-key" \
  --from-literal=ELEVENLABS_API_KEY="your-key" \
  --namespace=relicon-ai

# 3. Deploy using Helm chart (you'll need to create this)
helm install relicon-ai ./k8s/helm-chart \
  --namespace=relicon-ai \
  --values=./k8s/production-values.yaml
```

### Option 3: Cloud Platform Deployment

#### AWS Deployment
```bash
# Using AWS Copilot
copilot app init relicon-ai
copilot env init --name production
copilot svc init --name backend --svc-type "Backend Service"
copilot svc init --name frontend --svc-type "Load Balanced Web Service"
copilot env deploy --name production
copilot svc deploy --name backend --env production
copilot svc deploy --name frontend --env production
```

#### Google Cloud Platform
```bash
# Using Cloud Run
gcloud config set project YOUR_PROJECT_ID

# Deploy backend
gcloud run deploy relicon-backend \
  --source ./backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=$DATABASE_URL,OPENAI_API_KEY=$OPENAI_API_KEY

# Deploy frontend
gcloud run deploy relicon-frontend \
  --source ./frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Production Configuration

### Environment Variables

```bash
# Production .env file
DATABASE_URL=postgresql://relicon_user:SECURE_PASSWORD@db.internal:5432/relicon_prod
REDIS_URL=redis://redis.internal:6379/0

# AI Service Keys
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
LUMA_API_KEY=luma_xxxxxxxxxxxxxxxxxxxxxxxx
ELEVENLABS_API_KEY=eleven_xxxxxxxxxxxxxxxxxxxxxxxx

# Security
SECRET_KEY=ultra-secure-key-minimum-64-characters-for-production-use-only
DEBUG=False

# Performance
WORKERS=4
HOST=0.0.0.0
PORT=8000

# CORS (restrict to your domains)
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# File Storage (use cloud storage in production)
UPLOAD_DIR=/app/data/uploads
OUTPUT_DIR=/app/data/outputs
MAX_FILE_SIZE=104857600

# Monitoring
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
LOG_LEVEL=INFO

# Database Connection Pool
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
```

### Nginx Reverse Proxy Configuration

```nginx
# /etc/nginx/sites-available/relicon-ai
upstream backend {
    server 127.0.0.1:8000;
    # Add more backend instances for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

upstream frontend {
    server 127.0.0.1:3000;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # API Routes
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
    
    # Static files (outputs)
    location /outputs/ {
        alias /app/data/outputs/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        
        # Security for generated videos
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
    }
    
    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Handle React Router
        try_files $uri $uri/ @fallback;
    }
    
    location @fallback {
        proxy_pass http://frontend;
    }
}
```

### SSL Certificate Setup

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Database Configuration

### PostgreSQL Production Setup

```sql
-- Create production database and user
CREATE DATABASE relicon_prod;
CREATE USER relicon_user WITH ENCRYPTED PASSWORD 'ultra_secure_password';
GRANT ALL PRIVILEGES ON DATABASE relicon_prod TO relicon_user;

-- Performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Restart PostgreSQL to apply changes
```

### Database Backup Strategy

```bash
#!/bin/bash
# backup-database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/relicon"
mkdir -p $BACKUP_DIR

# Create backup
pg_dump -h localhost -U relicon_user -d relicon_prod | gzip > $BACKUP_DIR/relicon_$DATE.sql.gz

# Keep only last 7 days of backups
find $BACKUP_DIR -name "relicon_*.sql.gz" -mtime +7 -delete

# Upload to cloud storage (optional)
# aws s3 cp $BACKUP_DIR/relicon_$DATE.sql.gz s3://your-backup-bucket/database/
```

```bash
# Add to crontab for daily backups
0 2 * * * /path/to/backup-database.sh
```

## Monitoring & Observability

### Application Metrics

```python
# Add to backend/main.py
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Metrics
REQUEST_COUNT = Counter('relicon_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('relicon_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(time.time() - start_time)
    return response

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Docker Compose with Monitoring

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # ... existing services ...
  
  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
  
  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  prometheus_data:
  grafana_data:
```

## Security Considerations

### Application Security

```python
# Security middleware (add to main.py)
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# Only allow specific hosts
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)

# Redirect HTTP to HTTPS
app.add_middleware(HTTPSRedirectMiddleware)

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/create-ad")
@limiter.limit("5/minute")  # 5 requests per minute per IP
async def create_ad(request: Request, ...):
    # ... existing code ...
```

### Infrastructure Security

```bash
# Firewall configuration (UFW)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Docker security
# Run containers as non-root user
# Limit container resources
# Use security profiles
```

## Performance Optimization

### Backend Optimization

```python
# Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### Redis Configuration

```bash
# redis.conf optimizations
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Scaling Strategy

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  backend:
    image: relicon-backend
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1GB
        reservations:
          cpus: '0.5'
          memory: 512MB
  
  celery_worker:
    image: relicon-backend
    command: celery -A tasks worker --loglevel=info --concurrency=2
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2.0'
          memory: 2GB
```

### Auto-scaling (Kubernetes)

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: relicon-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: relicon-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# Full system backup script

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="/backups/$BACKUP_DATE"
mkdir -p $BACKUP_ROOT

# Database backup
pg_dump -h db.internal -U relicon_user relicon_prod | gzip > $BACKUP_ROOT/database.sql.gz

# Application data
tar -czf $BACKUP_ROOT/uploads.tar.gz /app/data/uploads/
tar -czf $BACKUP_ROOT/outputs.tar.gz /app/data/outputs/

# Configuration
cp -r /app/config $BACKUP_ROOT/
cp /app/.env $BACKUP_ROOT/

# Upload to cloud storage
aws s3 sync $BACKUP_ROOT s3://relicon-backups/$BACKUP_DATE/

# Cleanup local backups older than 7 days
find /backups -type d -mtime +7 -exec rm -rf {} +
```

### Recovery Procedures

```bash
# Database recovery
gunzip -c database.sql.gz | psql -h db.internal -U relicon_user relicon_prod

# Application data recovery
tar -xzf uploads.tar.gz -C /
tar -xzf outputs.tar.gz -C /

# Restart services
docker-compose restart
```

## Cost Optimization

### AI Service Cost Management

```python
# Cost tracking middleware
import asyncio
from datetime import datetime

cost_tracker = {
    "openai": 0.0,
    "luma": 0.0,
    "elevenlabs": 0.0
}

async def track_openai_cost(tokens_used: int):
    # Approximate cost calculation
    cost_per_1k_tokens = 0.002  # Update based on current pricing
    cost = (tokens_used / 1000) * cost_per_1k_tokens
    cost_tracker["openai"] += cost
    
    # Alert if costs are high
    if cost_tracker["openai"] > 100:  # $100 threshold
        # Send alert to admin
        pass
```

### Resource Optimization

```bash
# Docker resource limits
docker run --memory="1g" --cpus="1.0" relicon-backend

# Container auto-restart on high memory usage
docker run --restart=unless-stopped --oom-kill-disable=false relicon-backend
```

## Go-Live Checklist

### Pre-Launch
- [ ] All tests pass in production environment
- [ ] SSL certificates installed and working
- [ ] Domain DNS configured correctly
- [ ] Database optimized and backed up
- [ ] Monitoring and alerts configured
- [ ] Security measures implemented
- [ ] Performance baseline established

### Launch Day
- [ ] Deploy to production during low-traffic hours
- [ ] Monitor system resources closely
- [ ] Check all endpoints respond correctly
- [ ] Verify payment/billing systems (if applicable)
- [ ] Monitor error rates and response times
- [ ] Have rollback plan ready

### Post-Launch
- [ ] Monitor for 24-48 hours continuously
- [ ] Collect user feedback
- [ ] Review performance metrics
- [ ] Plan scaling based on actual usage
- [ ] Document any issues and resolutions

---

## 🎉 Production Success!

Your revolutionary AI ad creation platform is now live and ready to change the world of advertising!

**Key Production URLs:**
- Main App: https://yourdomain.com
- API Health: https://yourdomain.com/api/health
- Admin Dashboard: https://yourdomain.com/admin (if implemented)
- Monitoring: https://yourdomain.com:3001 (Grafana)

**Support & Maintenance:**
- Monitor logs daily
- Review performance metrics weekly
- Update dependencies monthly
- Backup verification monthly
- Security audit quarterly

**Ready to revolutionize advertising!** 🚀✨ 