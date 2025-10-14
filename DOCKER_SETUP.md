# Docker Setup Guide

This guide explains how to run the Maternal Healthcare Backend using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start

### 1. Clone and Navigate to Project

```bash
cd /home/mohamedalrawy/Desktop/projects/maternal/maternal_backend
```

### 2. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` and update the following:
- `SECRET_KEY`: Generate a new secret key for production
- `DB_PASSWORD`: Set a strong PostgreSQL password
- `ALLOWED_HOSTS`: Add your domain name

### 3. Build and Start Services

```bash
docker-compose up -d --build
```

This will start:
- **PostgreSQL** (port 5432)
- **Django Backend** (port 8003)
- **Nginx** (port 8080)
- **Nginx Proxy Manager** (ports 80, 443, 81)

### 4. Access Services

- **API**: http://localhost:8003
- **Nginx (Static files)**: http://localhost:8080
- **Nginx Proxy Manager Admin**: http://localhost:81
  - Default login: `admin@example.com` / `changeme`

### 5. Create Superuser

```bash
docker-compose exec web python manage.py createsuperuser
```

### 6. Load Initial Data (Optional)

```bash
docker-compose exec web python manage.py load_medical_qa
```

## Docker Compose Services

### 1. Database (PostgreSQL)
- **Container**: `maternal_db`
- **Port**: 5432
- **Volume**: `postgres_data`
- **Health Check**: Enabled

### 2. Django Backend
- **Container**: `maternal_backend`
- **Port**: 8003
- **Command**: Gunicorn with 3 workers
- **Volumes**: 
  - Code: `.:/app`
  - Static files: `static_volume`
  - Media files: `media_volume`

### 3. Nginx (Static Files)
- **Container**: `maternal_nginx`
- **Port**: 8080
- **Purpose**: Serves static and media files

### 4. Nginx Proxy Manager
- **Container**: `nginx_proxy_manager`
- **Ports**: 
  - 80 (HTTP)
  - 443 (HTTPS)
  - 81 (Admin UI)
- **Purpose**: Reverse proxy with SSL/TLS

## Common Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f web
docker-compose logs -f db
```

### Run Django Commands
```bash
# Migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Shell
docker-compose exec web python manage.py shell

# Collect static files
docker-compose exec web python manage.py collectstatic --noinput
```

### Database Access
```bash
# PostgreSQL shell
docker-compose exec db psql -U postgres -d maternal_db

# Backup database
docker-compose exec db pg_dump -U postgres maternal_db > backup.sql

# Restore database
docker-compose exec -T db psql -U postgres maternal_db < backup.sql
```

### Rebuild After Code Changes
```bash
docker-compose up -d --build
```

### Clean Everything
```bash
# Stop and remove containers, networks
docker-compose down

# Remove volumes (WARNING: Deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Nginx Proxy Manager Setup

### 1. Access Admin Panel
- URL: http://localhost:81
- Default credentials: `admin@example.com` / `changeme`

### 2. Create Proxy Host

1. Go to **Proxy Hosts** → **Add Proxy Host**
2. Fill in the details:
   - **Domain Names**: `api.yourdomain.com`
   - **Scheme**: `http`
   - **Forward Hostname/IP**: `maternal_backend` (or `web`)
   - **Forward Port**: `8003`
   - **Cache Assets**: Enabled
   - **Block Common Exploits**: Enabled
   - **Websockets Support**: Enabled

3. **SSL** tab:
   - Enable **Force SSL**
   - Enable **HTTP/2 Support**
   - Request SSL Certificate (Let's Encrypt)

4. Click **Save**

### 3. Access Your API
- Production: `https://api.yourdomain.com`
- Development: `http://localhost:8003`

## Environment Variables

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | Required |
| `DEBUG` | Debug mode | `True` |
| `DB_NAME` | Database name | `maternal_db` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `postgres` |
| `DB_HOST` | Database host | `db` |
| `DB_PORT` | Database port | `5432` |
| `ALLOWED_HOSTS` | Allowed hosts | `localhost,127.0.0.1` |

## Troubleshooting

### Database Connection Issues
```bash
# Check if database is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Test connection
docker-compose exec web python manage.py dbshell
```

### Static Files Not Loading
```bash
# Collect static files
docker-compose exec web python manage.py collectstatic --noinput

# Check nginx logs
docker-compose logs nginx
```

### Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :8003
sudo lsof -i :5432

# Change port in docker-compose.yml
ports:
  - "8004:8003"  # Change external port
```

### Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Fix entrypoint script
chmod +x entrypoint.sh
```

## Production Deployment

### Security Checklist

- [ ] Change `SECRET_KEY` in `.env`
- [ ] Set `DEBUG=False`
- [ ] Update `ALLOWED_HOSTS` with your domain
- [ ] Use strong database passwords
- [ ] Enable SSL/TLS via Nginx Proxy Manager
- [ ] Set up firewall rules
- [ ] Enable database backups
- [ ] Configure log rotation
- [ ] Set up monitoring
- [ ] Enable rate limiting

### Backup Strategy

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T db pg_dump -U postgres maternal_db | gzip > backup_$DATE.sql.gz

# Keep only last 7 days
find . -name "backup_*.sql.gz" -mtime +7 -delete
```

### Monitoring

```bash
# Check container status
docker-compose ps

# Resource usage
docker stats

# Logs
docker-compose logs --tail=100 -f
```

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review documentation: `README.md`
- Contact: Qassim Health Cluster

## License

© 2024 Qassim Health Cluster. All rights reserved.

