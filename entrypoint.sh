#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
while ! nc -z $DB_HOST $DB_PORT; do
  sleep 0.1
done
echo "PostgreSQL is up!"

# Run migrations
echo "Running migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear || true

# Load initial data if needed (optional)
# echo "Loading initial data..."
# python manage.py load_medical_qa || true

# Create superuser if it doesn't exist (optional)
# echo "Creating superuser..."
# python manage.py shell << EOF
# from django.contrib.auth import get_user_model
# User = get_user_model()
# if not User.objects.filter(username='admin').exists():
#     User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
# EOF

echo "Starting server..."
exec "$@"

