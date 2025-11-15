#!/bin/bash
################################################################################
# Maternal Backend - Automated Secure Deployment Script
# This script automates the deployment process with security checks
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}==>${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    return 0
}

################################################################################
# Pre-flight Checks
################################################################################

preflight_checks() {
    log_step "Running pre-flight checks..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "python3")
    for cmd in "${required_commands[@]}"; do
        if check_command $cmd; then
            log_info "$cmd is installed ✓"
        else
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    log_info "Docker daemon is running ✓"
    
    # Check if .env exists
    if [ ! -f .env ]; then
        log_warn ".env file not found"
        if [ -f .env.example ]; then
            log_info "Copying .env.example to .env"
            cp .env.example .env
            log_warn "Please edit .env file with your configuration"
            exit 1
        else
            log_error ".env.example not found"
            exit 1
        fi
    fi
    log_info ".env file exists ✓"
    
    log_info "Pre-flight checks passed ✓"
}

################################################################################
# Security Check
################################################################################

security_check() {
    log_step "Running security checks..."
    
    if [ -f security_check.py ]; then
        python3 security_check.py
        if [ $? -ne 0 ]; then
            log_error "Security check failed"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        log_warn "security_check.py not found, skipping security checks"
    fi
}

################################################################################
# Setup Directories
################################################################################

setup_directories() {
    log_step "Setting up directories..."
    
    local dirs=("logs" "db_backups" "nginx_ssl" "certbot_webroot" "nginx_cache")
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 logs db_backups 2>/dev/null || true
    
    log_info "Directories setup complete ✓"
}

################################################################################
# SSL Certificate Setup
################################################################################

setup_ssl() {
    log_step "Setting up SSL certificates..."
    
    if [ -f nginx_ssl/fullchain.pem ] && [ -f nginx_ssl/privkey.pem ]; then
        log_info "SSL certificates already exist ✓"
        return
    fi
    
    log_warn "SSL certificates not found"
    echo "Choose SSL certificate option:"
    echo "1) Generate self-signed certificate (testing only)"
    echo "2) Use existing Let's Encrypt certificate"
    echo "3) Skip (configure manually later)"
    read -p "Enter choice [1-3]: " ssl_choice
    
    case $ssl_choice in
        1)
            log_info "Generating self-signed certificate..."
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout nginx_ssl/privkey.pem \
                -out nginx_ssl/fullchain.pem \
                -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
                2>/dev/null
            chmod 644 nginx_ssl/*.pem
            log_info "Self-signed certificate created ✓"
            ;;
        2)
            read -p "Enter domain name: " domain
            if [ -d "/etc/letsencrypt/live/$domain" ]; then
                sudo cp /etc/letsencrypt/live/$domain/fullchain.pem nginx_ssl/
                sudo cp /etc/letsencrypt/live/$domain/privkey.pem nginx_ssl/
                sudo chmod 644 nginx_ssl/*.pem
                log_info "Let's Encrypt certificates copied ✓"
            else
                log_error "Let's Encrypt certificates not found for $domain"
                exit 1
            fi
            ;;
        3)
            log_warn "Skipping SSL setup - configure manually before starting services"
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

################################################################################
# Build Application
################################################################################

build_application() {
    log_step "Building application..."
    
    # Ask if user wants to compile Python code
    read -p "Compile Python code to .pyc? (recommended for production) [Y/n]: " compile_choice
    
    if [[ ! $compile_choice =~ ^[Nn]$ ]]; then
        if [ -f build_secure.py ]; then
            log_info "Running code compilation..."
            python3 build_secure.py <<EOF
n
EOF
            log_info "Code compilation complete ✓"
        else
            log_warn "build_secure.py not found, skipping compilation"
        fi
    fi
    
    # Build Docker images
    log_info "Building Docker images..."
    docker-compose -f docker-compose.prod.yml build --no-cache
    
    if [ $? -eq 0 ]; then
        log_info "Docker images built successfully ✓"
    else
        log_error "Docker build failed"
        exit 1
    fi
}

################################################################################
# Deploy Services
################################################################################

deploy_services() {
    log_step "Deploying services..."
    
    # Start database first
    log_info "Starting database service..."
    docker-compose -f docker-compose.prod.yml up -d db redis
    
    # Wait for database
    log_info "Waiting for database to be ready..."
    sleep 10
    
    # Start all services
    log_info "Starting all services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    if [ $? -eq 0 ]; then
        log_info "Services deployed successfully ✓"
    else
        log_error "Service deployment failed"
        exit 1
    fi
}

################################################################################
# Post-deployment Tasks
################################################################################

post_deployment() {
    log_step "Running post-deployment tasks..."
    
    # Wait for services to start
    log_info "Waiting for services to be ready..."
    sleep 15
    
    # Check service status
    docker-compose -f docker-compose.prod.yml ps
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    sleep 5
    
    if curl -f -s http://localhost/health > /dev/null 2>&1; then
        log_info "Health check passed ✓"
    else
        log_warn "Health check failed - services may still be starting"
    fi
    
    # Ask about superuser creation
    read -p "Create Django superuser? [y/N]: " superuser_choice
    
    if [[ $superuser_choice =~ ^[Yy]$ ]]; then
        docker-compose -f docker-compose.prod.yml exec web python manage.py createsuperuser
    fi
    
    log_info "Post-deployment tasks complete ✓"
}

################################################################################
# Display Summary
################################################################################

display_summary() {
    echo -e "\n${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         🎉 DEPLOYMENT SUCCESSFUL 🎉                      ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n${BLUE}Service Status:${NC}"
    docker-compose -f docker-compose.prod.yml ps
    
    echo -e "\n${BLUE}Access Points:${NC}"
    echo "  • Health Check: http://localhost/health"
    echo "  • API: https://localhost/api/"
    echo "  • Admin: https://localhost/admin/"
    
    echo -e "\n${BLUE}Useful Commands:${NC}"
    echo "  • View logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "  • Restart: docker-compose -f docker-compose.prod.yml restart"
    echo "  • Stop: docker-compose -f docker-compose.prod.yml down"
    echo "  • Check status: docker-compose -f docker-compose.prod.yml ps"
    
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo "  1. Update domain in nginx.prod.conf"
    echo "  2. Configure proper SSL certificates"
    echo "  3. Set up automated backups"
    echo "  4. Configure monitoring"
    echo "  5. Review security settings"
    
    echo -e "\n${YELLOW}Documentation:${NC}"
    echo "  • SECURE_DEPLOYMENT_GUIDE.md - Complete deployment guide"
    echo "  • QUICK_START.md - Quick reference"
    
    echo ""
}

################################################################################
# Rollback Function
################################################################################

rollback() {
    log_error "Deployment failed. Rolling back..."
    docker-compose -f docker-compose.prod.yml down
    log_info "Rollback complete"
    exit 1
}

################################################################################
# Main Deployment Flow
################################################################################

main() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║     Maternal Backend - Secure Deployment Script          ║
║     Automated Production Deployment                      ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    # Trap errors
    trap rollback ERR
    
    # Deployment steps
    preflight_checks
    security_check
    setup_directories
    setup_ssl
    build_application
    deploy_services
    post_deployment
    display_summary
}

################################################################################
# Entry Point
################################################################################

# Check if running with required permissions
if [ "$EUID" -eq 0 ]; then 
    log_warn "Running as root is not recommended"
    read -p "Continue as root? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run main deployment
main

exit 0

