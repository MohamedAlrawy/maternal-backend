#!/bin/bash
################################################################################
# Maternal Backend - Deployment Management Script
# Quick management commands for common operations
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

COMPOSE_FILE="docker-compose.prod.yml"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_menu() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║     Maternal Backend - Deployment Manager               ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo "1)  Start all services"
    echo "2)  Stop all services"
    echo "3)  Restart all services"
    echo "4)  View logs (all)"
    echo "5)  View logs (web only)"
    echo "6)  View logs (db only)"
    echo "7)  Check service status"
    echo "8)  Run database migrations"
    echo "9)  Create superuser"
    echo "10) Backup database"
    echo "11) Restore database"
    echo "12) Collect static files"
    echo "13) Django shell"
    echo "14) Container shell (web)"
    echo "15) Database shell"
    echo "16) Update application"
    echo "17) Security check"
    echo "18) Health check"
    echo "19) Clean up (remove volumes)"
    echo "20) View container stats"
    echo "0)  Exit"
    echo ""
}

################################################################################
# Service Management
################################################################################

start_services() {
    log_info "Starting all services..."
    docker-compose -f $COMPOSE_FILE up -d
    log_info "Services started. Check status with option 7"
}

stop_services() {
    log_info "Stopping all services..."
    docker-compose -f $COMPOSE_FILE down
    log_info "Services stopped"
}

restart_services() {
    log_info "Restarting all services..."
    docker-compose -f $COMPOSE_FILE restart
    log_info "Services restarted"
}

################################################################################
# Logging
################################################################################

view_logs_all() {
    log_info "Viewing all logs (Ctrl+C to exit)..."
    docker-compose -f $COMPOSE_FILE logs -f
}

view_logs_web() {
    log_info "Viewing web logs (Ctrl+C to exit)..."
    docker-compose -f $COMPOSE_FILE logs -f web
}

view_logs_db() {
    log_info "Viewing database logs (Ctrl+C to exit)..."
    docker-compose -f $COMPOSE_FILE logs -f db
}

################################################################################
# Status & Health
################################################################################

check_status() {
    log_info "Service status:"
    docker-compose -f $COMPOSE_FILE ps
}

health_check() {
    log_info "Running health checks..."
    
    echo -e "\n${BLUE}1. HTTP Health Check:${NC}"
    if curl -f -s http://localhost/health > /dev/null 2>&1; then
        log_info "✓ Health endpoint accessible"
        curl -s http://localhost/health | python3 -m json.tool 2>/dev/null || echo "OK"
    else
        log_error "✗ Health endpoint not accessible"
    fi
    
    echo -e "\n${BLUE}2. Container Status:${NC}"
    docker-compose -f $COMPOSE_FILE ps
    
    echo -e "\n${BLUE}3. Database Connectivity:${NC}"
    if docker-compose -f $COMPOSE_FILE exec -T db pg_isready > /dev/null 2>&1; then
        log_info "✓ Database is ready"
    else
        log_error "✗ Database is not ready"
    fi
    
    echo -e "\n${BLUE}4. Redis Connectivity:${NC}"
    if docker-compose -f $COMPOSE_FILE exec -T redis redis-cli ping > /dev/null 2>&1; then
        log_info "✓ Redis is ready"
    else
        log_error "✗ Redis is not ready"
    fi
}

container_stats() {
    log_info "Container resource usage:"
    docker stats --no-stream
}

################################################################################
# Database Operations
################################################################################

run_migrations() {
    log_info "Running database migrations..."
    docker-compose -f $COMPOSE_FILE exec web python manage.py migrate
    log_info "Migrations complete"
}

create_superuser() {
    log_info "Creating Django superuser..."
    docker-compose -f $COMPOSE_FILE exec web python manage.py createsuperuser
}

backup_database() {
    local backup_dir="db_backups"
    mkdir -p $backup_dir
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$backup_dir/backup_$timestamp.sql"
    
    log_info "Backing up database to $backup_file..."
    
    docker-compose -f $COMPOSE_FILE exec -T db pg_dump \
        -U ${DB_USER:-maternal_user} \
        ${DB_NAME:-maternal_prod} > $backup_file
    
    # Compress backup
    gzip $backup_file
    
    log_info "Backup complete: ${backup_file}.gz"
    
    # Show backup size
    ls -lh "${backup_file}.gz"
}

restore_database() {
    local backup_dir="db_backups"
    
    echo "Available backups:"
    ls -lh $backup_dir/*.sql* 2>/dev/null || echo "No backups found"
    echo ""
    
    read -p "Enter backup filename: " backup_file
    
    if [ ! -f "$backup_dir/$backup_file" ]; then
        log_error "Backup file not found"
        return
    fi
    
    log_error "WARNING: This will overwrite the current database!"
    read -p "Are you sure? (type 'yes' to continue): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Restore cancelled"
        return
    fi
    
    log_info "Restoring database from $backup_file..."
    
    # Check if file is gzipped
    if [[ $backup_file == *.gz ]]; then
        gunzip -c "$backup_dir/$backup_file" | \
            docker-compose -f $COMPOSE_FILE exec -T db psql \
            -U ${DB_USER:-maternal_user} \
            -d ${DB_NAME:-maternal_prod}
    else
        docker-compose -f $COMPOSE_FILE exec -T db psql \
            -U ${DB_USER:-maternal_user} \
            -d ${DB_NAME:-maternal_prod} < "$backup_dir/$backup_file"
    fi
    
    log_info "Database restored"
}

database_shell() {
    log_info "Opening database shell..."
    docker-compose -f $COMPOSE_FILE exec db psql \
        -U ${DB_USER:-maternal_user} \
        -d ${DB_NAME:-maternal_prod}
}

################################################################################
# Application Operations
################################################################################

collect_static() {
    log_info "Collecting static files..."
    docker-compose -f $COMPOSE_FILE exec web python manage.py collectstatic --noinput
    log_info "Static files collected"
}

django_shell() {
    log_info "Opening Django shell..."
    docker-compose -f $COMPOSE_FILE exec web python manage.py shell
}

container_shell() {
    log_info "Opening web container shell..."
    docker-compose -f $COMPOSE_FILE exec web bash
}

update_application() {
    log_info "Updating application..."
    
    echo "This will:"
    echo "  1. Pull latest code"
    echo "  2. Rebuild Docker images"
    echo "  3. Run migrations"
    echo "  4. Restart services"
    echo ""
    read -p "Continue? [y/N]: " confirm
    
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        log_info "Update cancelled"
        return
    fi
    
    log_info "Pulling latest code..."
    git pull origin main || log_error "Git pull failed"
    
    log_info "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --no-cache
    
    log_info "Running migrations..."
    docker-compose -f $COMPOSE_FILE exec web python manage.py migrate
    
    log_info "Collecting static files..."
    docker-compose -f $COMPOSE_FILE exec web python manage.py collectstatic --noinput
    
    log_info "Restarting services..."
    docker-compose -f $COMPOSE_FILE restart
    
    log_info "Update complete!"
}

################################################################################
# Security & Cleanup
################################################################################

run_security_check() {
    if [ -f security_check.py ]; then
        log_info "Running security check..."
        python3 security_check.py
    else
        log_error "security_check.py not found"
    fi
}

cleanup() {
    log_error "WARNING: This will remove all containers, volumes, and data!"
    read -p "Are you sure? (type 'yes' to continue): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_info "Cleanup cancelled"
        return
    fi
    
    log_info "Stopping and removing containers, volumes..."
    docker-compose -f $COMPOSE_FILE down -v
    
    log_info "Cleanup complete"
}

################################################################################
# Main Menu Loop
################################################################################

main() {
    while true; do
        show_menu
        read -p "Enter choice [0-20]: " choice
        echo ""
        
        case $choice in
            1) start_services ;;
            2) stop_services ;;
            3) restart_services ;;
            4) view_logs_all ;;
            5) view_logs_web ;;
            6) view_logs_db ;;
            7) check_status ;;
            8) run_migrations ;;
            9) create_superuser ;;
            10) backup_database ;;
            11) restore_database ;;
            12) collect_static ;;
            13) django_shell ;;
            14) container_shell ;;
            15) database_shell ;;
            16) update_application ;;
            17) run_security_check ;;
            18) health_check ;;
            19) cleanup ;;
            20) container_stats ;;
            0) log_info "Goodbye!"; exit 0 ;;
            *) log_error "Invalid choice" ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

################################################################################
# Entry Point
################################################################################

# Check if docker-compose file exists
if [ ! -f $COMPOSE_FILE ]; then
    log_error "docker-compose.prod.yml not found in current directory"
    exit 1
fi

# Run main menu
main

exit 0

