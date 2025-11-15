#!/bin/bash
# =============================================================================
# SECURE PRODUCTION ENTRYPOINT SCRIPT
# Handles database initialization, migrations, and health checks
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Security: Validate required environment variables
REQUIRED_VARS=("DB_NAME" "DB_USER" "DB_PASSWORD" "DB_HOST" "SECRET_KEY")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        log_error "Required environment variable $var is not set!"
        exit 1
    fi
done

log_info "Starting Maternal Backend (Production)"
log_info "Python version: $(python --version)"
log_info "Django environment: ${DJANGO_SETTINGS_MODULE:-maternal_backend.settings}"

# Wait for PostgreSQL to be ready
log_info "Waiting for PostgreSQL to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

until PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        log_error "PostgreSQL did not become ready in time!"
        exit 1
    fi
    log_warn "PostgreSQL is unavailable - attempt $RETRY_COUNT/$MAX_RETRIES"
    sleep 2
done

log_info "PostgreSQL is ready!"

# Run database migrations
log_info "Running database migrations..."
if python manage.py migrate --noinput; then
    log_info "Migrations completed successfully"
else
    log_error "Migration failed!"
    exit 1
fi

# Collect static files
log_info "Collecting static files..."
if python manage.py collectstatic --noinput --clear; then
    log_info "Static files collected successfully"
else
    log_warn "Static files collection failed (non-critical)"
fi

# Create cache tables if using database cache
log_info "Creating cache tables..."
python manage.py createcachetable 2>/dev/null || log_warn "Cache table creation skipped"

# Check database connectivity
log_info "Checking database connectivity..."
if python manage.py check --database default; then
    log_info "Database connectivity check passed"
else
    log_error "Database connectivity check failed!"
    exit 1
fi

# Security: Check for deployment issues
log_info "Running deployment security checks..."
python manage.py check --deploy --fail-level WARNING 2>/dev/null || log_warn "Some security checks failed (review logs)"

# Optional: Load initial data for first deployment
if [ "$LOAD_INITIAL_DATA" = "true" ]; then
    log_info "Loading initial data..."
    python manage.py load_medical_qa || log_warn "Initial data loading failed (non-critical)"
fi

# Optional: Create superuser if specified
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
    log_info "Creating superuser..."
    python manage.py shell << EOF || true
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='$DJANGO_SUPERUSER_USERNAME').exists():
    User.objects.create_superuser(
        username='$DJANGO_SUPERUSER_USERNAME',
        email='${DJANGO_SUPERUSER_EMAIL:-admin@example.com}',
        password='$DJANGO_SUPERUSER_PASSWORD'
    )
    print("Superuser created successfully")
else:
    print("Superuser already exists")
EOF
fi

# Set proper permissions
chmod -R 755 /app/staticfiles 2>/dev/null || true
chmod -R 755 /app/media 2>/dev/null || true

log_info "=== Initialization Complete ==="
log_info "Starting application server..."

# Execute the main command
exec "$@"

