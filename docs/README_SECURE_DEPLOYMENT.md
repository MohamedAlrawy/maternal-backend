# 🔐 Maternal Backend - Secure Production Deployment

[![Security](https://img.shields.io/badge/Security-Hardened-green.svg)](./SECURE_DEPLOYMENT_GUIDE.md)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](./docker-compose.prod.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.2.6-green.svg)](https://djangoproject.com)

## 🎯 Overview

This is a **production-ready, security-hardened** deployment system for the Maternal Backend application. It includes:

- 🔒 **Compiled Python code** (.pyc) for IP protection
- 🐳 **Multi-stage Docker builds** with security best practices
- 🔐 **SSL/TLS encryption** with modern ciphers
- 🛡️ **Rate limiting** and DDoS protection
- 📊 **Health monitoring** and logging
- 🚀 **Automated deployment** scripts
- 📦 **Database backups** and recovery
- 👤 **Non-root containers** for enhanced security

---

## ⚡ Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone <repo-url> && cd maternal_backend

# 2. Run automated deployment
./deploy.sh

# 3. Access your application
curl http://localhost/health
```

**That's it!** The script handles everything: security checks, SSL setup, building, and deployment.

---

## 📚 Documentation

### Main Guides

| Document | Description | Best For |
|----------|-------------|----------|
| [QUICK_START.md](./QUICK_START.md) | 5-minute setup guide | Getting started quickly |
| [SECURE_DEPLOYMENT_GUIDE.md](./SECURE_DEPLOYMENT_GUIDE.md) | Comprehensive deployment guide | Production deployment |
| [DEPLOYMENT_SUMMARY.md](./DEPLOYMENT_SUMMARY.md) | Architecture and features overview | Understanding the system |

### Scripts & Tools

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy.sh` | Automated deployment | `./deploy.sh` |
| `manage_deployment.sh` | Interactive management menu | `./manage_deployment.sh` |
| `security_check.py` | Pre-deployment security validation | `python3 security_check.py` |
| `build_secure.py` | Compile Python code to .pyc | `python3 build_secure.py` |

---

## 🏗️ Architecture

```
Internet → Nginx (SSL) → Django (Gunicorn) → PostgreSQL
                    ↓           ↓
              Static Files   Redis (Cache)
```

**Components:**
- **Nginx**: Reverse proxy, SSL termination, rate limiting
- **Django**: Application server (compiled .pyc code)
- **PostgreSQL**: Database with connection pooling
- **Redis**: Caching and session storage
- **Gunicorn**: WSGI server with 4 workers

---

## 🔒 Security Features

### Code Protection
✅ Python bytecode compilation (.pyc)  
✅ No source code in production image  
✅ Multi-stage Docker builds  

### Network Security
✅ HTTPS/TLS with modern ciphers  
✅ Rate limiting (API, auth endpoints)  
✅ CORS configuration  
✅ Security headers (HSTS, CSP, X-Frame-Options)  

### Application Security
✅ Non-root container user  
✅ Secure session management  
✅ Strong password validation  
✅ Secret key protection  
✅ Database connection pooling  

### Monitoring
✅ Health check endpoints  
✅ Structured logging  
✅ Error tracking support (Sentry)  
✅ Resource monitoring  

---

## 📋 Prerequisites

```bash
# Required
docker --version        # 20.10+
docker-compose --version # 2.0+
python3 --version       # 3.11+

# Optional
openssl version         # For SSL certificates
```

---

## 🚀 Deployment Options

### Option 1: Automated (Recommended)

```bash
./deploy.sh
```

**Includes:**
- ✅ Prerequisite checks
- ✅ Security validation
- ✅ SSL certificate setup
- ✅ Docker image building
- ✅ Service deployment
- ✅ Post-deployment verification

### Option 2: Interactive Management

```bash
./manage_deployment.sh
```

**Features:**
- Start/stop/restart services
- View logs
- Database operations (backup, restore, migrations)
- Health checks
- Update application
- And more...

### Option 3: Manual Deployment

```bash
# 1. Configure environment
cp .env.example .env
nano .env

# 2. Set up SSL
mkdir -p nginx_ssl
# Copy your SSL certificates to nginx_ssl/

# 3. Build and deploy
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# 4. Run migrations
docker-compose -f docker-compose.prod.yml exec web python manage.py migrate

# 5. Create superuser
docker-compose -f docker-compose.prod.yml exec web python manage.py createsuperuser
```

---

## 🔧 Common Commands

```bash
# View all logs
docker-compose -f docker-compose.prod.yml logs -f

# Check status
docker-compose -f docker-compose.prod.yml ps

# Restart application
docker-compose -f docker-compose.prod.yml restart web

# Database backup
docker-compose -f docker-compose.prod.yml exec db pg_dump \
  -U maternal_user maternal_prod > backup.sql

# Shell access
docker-compose -f docker-compose.prod.yml exec web bash

# Update application
git pull && docker-compose -f docker-compose.prod.yml up -d --build
```

---

## 🎨 Configuration

### Environment Variables

Key variables in `.env`:

```env
# Django
SECRET_KEY=<generate-secure-key>
DEBUG=False
DJANGO_ENV=production
ALLOWED_HOSTS=yourdomain.com

# Database
DB_NAME=maternal_prod
DB_USER=maternal_user
DB_PASSWORD=<strong-password>

# Redis
REDIS_PASSWORD=<strong-password>

# Security
SECURE_SSL_REDIRECT=True
SESSION_COOKIE_SECURE=True
```

Generate secure keys:
```bash
# SECRET_KEY
python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"

# Passwords
openssl rand -base64 32
```

### SSL Certificates

**Production (Let's Encrypt):**
```bash
certbot certonly --standalone -d yourdomain.com
cp /etc/letsencrypt/live/yourdomain.com/* nginx_ssl/
```

**Development (Self-signed):**
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx_ssl/privkey.pem \
  -out nginx_ssl/fullchain.pem
```

---

## 📊 Monitoring

### Health Checks

```bash
# HTTP health endpoint
curl http://localhost/health

# Container health
docker-compose -f docker-compose.prod.yml ps

# Database health
docker-compose -f docker-compose.prod.yml exec db pg_isready
```

### Logs

```bash
# Application logs
docker-compose -f docker-compose.prod.yml logs -f web

# Error logs (inside container)
docker-compose -f docker-compose.prod.yml exec web tail -f /app/logs/django_errors.log

# Security logs
docker-compose -f docker-compose.prod.yml exec web tail -f /app/logs/security.log
```

### Resource Monitoring

```bash
# Container stats
docker stats

# Disk usage
docker system df

# Database size
docker-compose -f docker-compose.prod.yml exec db \
  psql -U maternal_user -d maternal_prod \
  -c "SELECT pg_size_pretty(pg_database_size('maternal_prod'));"
```

---

## 🔄 Maintenance

### Database Backups

**Manual Backup:**
```bash
./manage_deployment.sh  # Option 10: Backup database
```

**Automated Backup (Cron):**
```bash
# Add to crontab
0 2 * * * /opt/maternal-backend/maternal_backend/manage_deployment.sh backup
```

### Updates

```bash
# Using management script
./manage_deployment.sh  # Option 16: Update application

# Or manually
git pull origin main
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml exec web python manage.py migrate
docker-compose -f docker-compose.prod.yml restart
```

### Scaling

```bash
# Scale to 3 web workers
docker-compose -f docker-compose.prod.yml up -d --scale web=3

# Update nginx configuration for load balancing
```

---

## 🆘 Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs

# Check specific service
docker-compose -f docker-compose.prod.yml logs web

# Recreate containers
docker-compose -f docker-compose.prod.yml up -d --force-recreate
```

### Database Connection Issues

```bash
# Test database connectivity
docker-compose -f docker-compose.prod.yml exec web python manage.py dbshell

# Check environment variables
docker-compose -f docker-compose.prod.yml exec web env | grep DB
```

### SSL Certificate Issues

```bash
# Check certificate validity
openssl x509 -in nginx_ssl/fullchain.pem -noout -dates

# Test SSL
curl -vI https://yourdomain.com

# Check nginx configuration
docker-compose -f docker-compose.prod.yml exec nginx nginx -t
```

### Permission Issues

```bash
# Fix ownership
docker-compose -f docker-compose.prod.yml exec web chown -R maternal:maternal /app

# Recreate volumes
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🔐 Security Checklist

Before production deployment:

- [ ] Strong `SECRET_KEY` generated
- [ ] `DEBUG=False` set
- [ ] Proper `ALLOWED_HOSTS` configured
- [ ] Strong database passwords
- [ ] SSL certificates installed
- [ ] Firewall configured (only 22, 80, 443 open)
- [ ] SSH key authentication enabled
- [ ] Regular backups configured
- [ ] Monitoring/alerting set up
- [ ] Security headers verified
- [ ] Rate limiting tested
- [ ] Admin panel access restricted

Run security check:
```bash
python3 security_check.py
```

---

## 📈 Performance

**Optimizations Included:**
- Gunicorn with 4 workers
- Database connection pooling
- Redis caching
- Nginx response caching
- Gzip compression
- Static file optimization
- Keep-alive connections

**Monitoring Performance:**
```bash
# Container resources
docker stats

# Application metrics (inside container)
docker-compose -f docker-compose.prod.yml exec web python manage.py shell
>>> from django.db import connection
>>> print(connection.queries)
```

---

## 🌐 Production Deployment

### Recommended Server Specs

**Minimum:**
- 2 CPU cores
- 4GB RAM
- 20GB disk
- Ubuntu 20.04+

**Recommended:**
- 4 CPU cores
- 8GB RAM
- 50GB disk
- Ubuntu 22.04 LTS

### Firewall Setup

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Verify
sudo ufw status
```

### Domain Configuration

1. Point your domain A record to server IP
2. Update `nginx.prod.conf` with your domain
3. Obtain Let's Encrypt certificate
4. Update `.env` with your domain

---

## 📞 Support & Resources

### Documentation
- Full Deployment Guide: [SECURE_DEPLOYMENT_GUIDE.md](./SECURE_DEPLOYMENT_GUIDE.md)
- Quick Start: [QUICK_START.md](./QUICK_START.md)
- Architecture Overview: [DEPLOYMENT_SUMMARY.md](./DEPLOYMENT_SUMMARY.md)

### External Resources
- [Django Deployment](https://docs.djangoproject.com/en/stable/howto/deployment/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Nginx Security](https://nginx.org/en/docs/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

### Getting Help

1. Check logs: `docker-compose -f docker-compose.prod.yml logs -f`
2. Run health check: `curl http://localhost/health`
3. Run security check: `python3 security_check.py`
4. Review troubleshooting section above

---

## 📝 License

Copyright © 2025 Maternal Health Team. All rights reserved.

---

## 🎉 Next Steps

After successful deployment:

1. ✅ Verify health checks pass
2. ✅ Test API endpoints
3. ✅ Configure monitoring
4. ✅ Set up automated backups
5. ✅ Review security settings
6. ✅ Configure domain and SSL
7. ✅ Set up log aggregation
8. ✅ Performance tuning

**Congratulations on your secure deployment! 🚀**

For detailed information, see [SECURE_DEPLOYMENT_GUIDE.md](./SECURE_DEPLOYMENT_GUIDE.md)

