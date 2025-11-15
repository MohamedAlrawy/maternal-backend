# ⚡ Quick Start - Secure Production Deployment

## Fast Track Deployment (5 Minutes)

### Prerequisites Check

```bash
# Verify installations
docker --version       # Should be 20.10+
docker-compose --version  # Should be 2.0+
```

### 1️⃣ Clone & Setup (1 min)

```bash
cd /opt
git clone <your-repo> maternal-backend
cd maternal-backend/maternal_backend
mkdir -p logs db_backups nginx_ssl
```

### 2️⃣ Configure Environment (2 min)

```bash
# Create .env file
cat > .env << 'EOF'
# Django
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(50))")
DJANGO_ENV=production
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1,yourdomain.com

# Database
DB_NAME=maternal_prod
DB_USER=maternal_user
DB_PASSWORD=$(openssl rand -base64 32)
DB_HOST=db
DB_PORT=5432

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=$(openssl rand -base64 32)

# CORS
CORS_ALLOWED_ORIGINS=https://yourdomain.com
CSRF_TRUSTED_ORIGINS=https://yourdomain.com
EOF

# Generate actual values
python3 << 'PYEOF'
import secrets
import subprocess

# Generate SECRET_KEY
secret_key = secrets.token_urlsafe(50)
print(f"SECRET_KEY={secret_key}")

# Generate passwords
db_pass = secrets.token_urlsafe(32)
redis_pass = secrets.token_urlsafe(32)
print(f"DB_PASSWORD={db_pass}")
print(f"REDIS_PASSWORD={redis_pass}")
PYEOF
```

**Copy the output and update your .env file**

### 3️⃣ SSL Certificate (1 min)

**For Testing (Self-Signed):**
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ./nginx_ssl/privkey.pem \
  -out ./nginx_ssl/fullchain.pem \
  -subj "/CN=localhost"
```

**For Production (Let's Encrypt):**
```bash
sudo certbot certonly --standalone -d yourdomain.com
sudo cp /etc/letsencrypt/live/yourdomain.com/* ./nginx_ssl/
sudo chmod 644 ./nginx_ssl/*
```

### 4️⃣ Update Configuration (30 sec)

```bash
# Update nginx domain
sed -i 's/your-domain.com/yourdomain.com/g' nginx.prod.conf
```

### 5️⃣ Deploy (1 min)

```bash
# Build and start
docker-compose -f docker-compose.prod.yml up -d --build

# Watch logs
docker-compose -f docker-compose.prod.yml logs -f
```

### 6️⃣ Verify (30 sec)

```bash
# Check health
curl http://localhost/health

# Check all services
docker-compose -f docker-compose.prod.yml ps
```

### 7️⃣ Create Admin User

```bash
docker-compose -f docker-compose.prod.yml exec web python manage.py createsuperuser
```

---

## ✅ Post-Deployment Checklist

- [ ] Health check passes: `curl http://localhost/health`
- [ ] All containers running: `docker-compose -f docker-compose.prod.yml ps`
- [ ] Admin accessible: `https://yourdomain.com/admin/`
- [ ] API responding: `https://yourdomain.com/api/`
- [ ] SSL working: `curl -I https://yourdomain.com`
- [ ] Database accessible from app
- [ ] Redis accessible from app
- [ ] Logs are being written: `ls -lh logs/`

---

## 🔧 Common Commands

```bash
# View logs
docker-compose -f docker-compose.prod.yml logs -f web

# Restart application
docker-compose -f docker-compose.prod.yml restart web

# Stop all
docker-compose -f docker-compose.prod.yml down

# Update application
git pull && docker-compose -f docker-compose.prod.yml up -d --build

# Backup database
docker-compose -f docker-compose.prod.yml exec db pg_dump -U maternal_user maternal_prod > backup.sql

# Shell access
docker-compose -f docker-compose.prod.yml exec web bash
```

---

## 🆘 Quick Troubleshooting

### Container won't start?
```bash
docker-compose -f docker-compose.prod.yml logs web
```

### Database issues?
```bash
docker-compose -f docker-compose.prod.yml logs db
docker-compose -f docker-compose.prod.yml restart db
```

### SSL errors?
```bash
# Check certificates
ls -l nginx_ssl/
openssl x509 -in nginx_ssl/fullchain.pem -noout -dates
```

### Permission issues?
```bash
docker-compose -f docker-compose.prod.yml exec web chown -R maternal:maternal /app
```

---

## 📚 Full Documentation

See [SECURE_DEPLOYMENT_GUIDE.md](./SECURE_DEPLOYMENT_GUIDE.md) for comprehensive documentation.

---

**Need Help?** Check logs first: `docker-compose -f docker-compose.prod.yml logs -f`

