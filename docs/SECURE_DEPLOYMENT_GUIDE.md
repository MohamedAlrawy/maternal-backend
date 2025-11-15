# 🔒 Maternal Backend - Secure Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Maternal Backend application in a secure production environment with compiled Python code for intellectual property protection.

## 🎯 Security Features

### Code Protection
- ✅ **Compiled Python files (.pyc)** - Source code is compiled to bytecode
- ✅ **Multi-stage Docker build** - Separate build and runtime environments
- ✅ **Non-root container user** - Enhanced container security
- ✅ **Read-only file systems** - Static files mounted as read-only

### Network Security
- ✅ **HTTPS/TLS encryption** - All traffic encrypted
- ✅ **Rate limiting** - Protection against brute force and DoS
- ✅ **CORS configuration** - Controlled cross-origin access
- ✅ **Security headers** - HSTS, CSP, X-Frame-Options, etc.
- ✅ **IP restriction** - Optional IP whitelisting for admin panel

### Application Security
- ✅ **Secret key management** - Environment-based configuration
- ✅ **Database connection pooling** - Optimized and secure connections
- ✅ **Session security** - Secure cookies and session management
- ✅ **Redis caching** - Password-protected Redis instance
- ✅ **Input validation** - File upload restrictions and validation

### Monitoring & Logging
- ✅ **Structured logging** - Comprehensive application logs
- ✅ **Health checks** - Container and application health monitoring
- ✅ **Error tracking** - Optional Sentry integration
- ✅ **Audit logging** - Security event logging

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Disk**: Minimum 20GB free space
- **CPU**: 2+ cores recommended

### Required Software
```bash
# Docker Engine 20.10+
docker --version

# Docker Compose 2.0+
docker-compose --version

# Git
git --version

# OpenSSL (for certificate generation)
openssl version
```

---

## 🚀 Step-by-Step Deployment

### Step 1: Clone and Prepare Repository

```bash
# Clone the repository
cd /opt
git clone <your-repo-url> maternal-backend
cd maternal-backend/maternal_backend

# Create necessary directories
mkdir -p logs db_backups nginx_ssl certbot_webroot nginx_cache
chmod 755 logs db_backups
```

### Step 2: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit environment file with secure values
nano .env
```

**Critical Environment Variables:**

```bash
# Generate a strong SECRET_KEY
python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"

# Generate strong passwords
openssl rand -base64 32
```

**Minimum Required Variables:**
```env
SECRET_KEY=<generated-secret-key>
DJANGO_ENV=production
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
DB_NAME=maternal_prod
DB_USER=maternal_user
DB_PASSWORD=<strong-password>
REDIS_PASSWORD=<strong-password>
```

### Step 3: SSL Certificate Setup

#### Option A: Let's Encrypt (Recommended for Production)

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot

# Obtain certificate (make sure domain points to your server)
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Copy certificates to project directory
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./nginx_ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./nginx_ssl/
sudo chmod 644 ./nginx_ssl/*.pem
```

#### Option B: Self-Signed Certificate (Development/Testing Only)

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ./nginx_ssl/privkey.pem \
  -out ./nginx_ssl/fullchain.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=yourdomain.com"

chmod 644 ./nginx_ssl/*.pem
```

### Step 4: Update Nginx Configuration

```bash
# Edit nginx.prod.conf
nano nginx.prod.conf

# Update server_name with your actual domain
# server_name yourdomain.com www.yourdomain.com;

# Verify SSL certificate paths
# ssl_certificate /etc/nginx/ssl/fullchain.pem;
# ssl_certificate_key /etc/nginx/ssl/privkey.pem;
```

### Step 5: Build Compiled Version (Optional)

To compile Python source code for additional protection:

```bash
# Install Python if not already installed
python3 -m pip install --upgrade pip

# Run the build script
python3 build_secure.py

# This creates a build_secure/ directory with compiled .pyc files
```

### Step 6: Build Docker Images

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build --no-cache

# Verify images are created
docker images | grep maternal
```

### Step 7: Initialize Database

```bash
# Start only the database first
docker-compose -f docker-compose.prod.yml up -d db

# Wait for database to be ready (check logs)
docker-compose -f docker-compose.prod.yml logs -f db

# When ready, press Ctrl+C to stop following logs
```

### Step 8: Deploy Application

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Monitor startup logs
docker-compose -f docker-compose.prod.yml logs -f web

# Check service status
docker-compose -f docker-compose.prod.yml ps
```

### Step 9: Create Superuser

```bash
# Access the web container
docker-compose -f docker-compose.prod.yml exec web bash

# Inside container, create superuser
python manage.py createsuperuser

# Exit container
exit
```

### Step 10: Verify Deployment

```bash
# Check health endpoint
curl http://localhost/health

# Check HTTPS (replace with your domain)
curl -k https://yourdomain.com/api/health/

# Check all containers are running
docker-compose -f docker-compose.prod.yml ps
```

---

## 🔐 Security Hardening Checklist

### Pre-Deployment

- [ ] Generate strong `SECRET_KEY`
- [ ] Set `DEBUG=False`
- [ ] Configure proper `ALLOWED_HOSTS`
- [ ] Set strong database passwords
- [ ] Configure Redis password
- [ ] Set up SSL certificates
- [ ] Review and update CORS settings
- [ ] Configure email settings
- [ ] Set up backup strategy

### Post-Deployment

- [ ] Change all default passwords
- [ ] Test SSL/TLS configuration
- [ ] Verify rate limiting works
- [ ] Test health check endpoints
- [ ] Configure firewall rules
- [ ] Set up monitoring/alerting
- [ ] Configure automated backups
- [ ] Review application logs
- [ ] Test disaster recovery procedure
- [ ] Document any customizations

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable

# Or iptables
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -j DROP
```

---

## 🔄 Maintenance Operations

### Viewing Logs

```bash
# All services
docker-compose -f docker-compose.prod.yml logs -f

# Specific service
docker-compose -f docker-compose.prod.yml logs -f web
docker-compose -f docker-compose.prod.yml logs -f db
docker-compose -f docker-compose.prod.yml logs -f nginx

# Application logs (inside container)
docker-compose -f docker-compose.prod.yml exec web tail -f /app/logs/django_errors.log
```

### Database Backup

```bash
# Manual backup
docker-compose -f docker-compose.prod.yml exec db pg_dump -U maternal_user maternal_prod > ./db_backups/backup_$(date +%Y%m%d_%H%M%S).sql

# Restore from backup
docker-compose -f docker-compose.prod.yml exec -T db psql -U maternal_user -d maternal_prod < ./db_backups/backup_20250101_120000.sql
```

### Automated Backup Script

```bash
# Create backup script
cat > backup_database.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/maternal-backend/maternal_backend/db_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

cd /opt/maternal-backend/maternal_backend
docker-compose -f docker-compose.prod.yml exec -T db pg_dump -U maternal_user maternal_prod | gzip > "$BACKUP_DIR/backup_$TIMESTAMP.sql.gz"

# Delete old backups
find "$BACKUP_DIR" -name "backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: backup_$TIMESTAMP.sql.gz"
EOF

chmod +x backup_database.sh

# Add to crontab
crontab -e
# Add line: 0 2 * * * /opt/maternal-backend/maternal_backend/backup_database.sh
```

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild images
docker-compose -f docker-compose.prod.yml build --no-cache

# Apply migrations
docker-compose -f docker-compose.prod.yml exec web python manage.py migrate

# Restart services
docker-compose -f docker-compose.prod.yml restart web

# Or full restart
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d
```

### Scale Application

```bash
# Scale web workers
docker-compose -f docker-compose.prod.yml up -d --scale web=3

# Update nginx to load balance (requires nginx configuration update)
```

---

## 📊 Monitoring

### Health Checks

```bash
# Application health
curl http://localhost/health

# Database health
docker-compose -f docker-compose.prod.yml exec db pg_isready -U maternal_user

# Redis health
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
```

### Resource Monitoring

```bash
# Container stats
docker stats

# Disk usage
docker system df

# Specific container
docker-compose -f docker-compose.prod.yml exec web df -h
```

### Log Monitoring (Optional - ELK Stack)

```yaml
# Add to docker-compose.prod.yml for centralized logging
  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
  
  kibana:
    image: kibana:8.8.0
    ports:
      - "5601:5601"
```

---

## 🆘 Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs web

# Check container status
docker-compose -f docker-compose.prod.yml ps

# Recreate container
docker-compose -f docker-compose.prod.yml up -d --force-recreate web
```

#### 2. Database Connection Issues

```bash
# Verify database is running
docker-compose -f docker-compose.prod.yml ps db

# Check database logs
docker-compose -f docker-compose.prod.yml logs db

# Test connection
docker-compose -f docker-compose.prod.yml exec web python manage.py dbshell
```

#### 3. SSL Certificate Issues

```bash
# Check certificate validity
openssl x509 -in ./nginx_ssl/fullchain.pem -noout -dates

# Test SSL configuration
curl -vI https://yourdomain.com

# Check nginx configuration
docker-compose -f docker-compose.prod.yml exec nginx nginx -t
```

#### 4. Permission Issues

```bash
# Fix ownership (inside container)
docker-compose -f docker-compose.prod.yml exec web chown -R maternal:maternal /app

# Fix static files
docker-compose -f docker-compose.prod.yml exec web python manage.py collectstatic --noinput
```

#### 5. Out of Memory

```bash
# Check memory usage
docker stats

# Increase container limits in docker-compose.prod.yml
# deploy:
#   resources:
#     limits:
#       memory: 4G
```

---

## 🔒 Additional Security Recommendations

### 1. Fail2Ban for SSH Protection

```bash
sudo apt-get install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 2. Regular Updates

```bash
# System updates
sudo apt-get update && sudo apt-get upgrade -y

# Docker updates
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Security Scanning

```bash
# Scan Docker images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image maternal_backend_prod:latest
```

### 4. Network Segmentation

```bash
# Create separate networks for different services
# Already configured in docker-compose.prod.yml
```

### 5. Secrets Management

Consider using Docker Secrets or external secret management:

```bash
# Docker Swarm Secrets
echo "my_secret_password" | docker secret create db_password -

# Or use HashiCorp Vault
# Or AWS Secrets Manager
```

---

## 📚 Additional Resources

- [Django Deployment Checklist](https://docs.djangoproject.com/en/stable/howto/deployment/checklist/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [PostgreSQL Security](https://www.postgresql.org/docs/current/security.html)

---

## 📞 Support

For issues or questions:
- Review logs: `docker-compose -f docker-compose.prod.yml logs`
- Check health: `curl http://localhost/health`
- Restart services: `docker-compose -f docker-compose.prod.yml restart`

---

## 📄 License

Copyright © 2025 Maternal Health Team. All rights reserved.

---

**Last Updated**: October 2025
**Version**: 1.0.0

