#!/usr/bin/env python3
"""
Security Check Script for Maternal Backend
Validates security configuration before deployment
"""
import os
import sys
from pathlib import Path


class SecurityChecker:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_failed = 0
        
    def check_env_file(self):
        """Check if .env file exists and has proper permissions"""
        print("\n🔍 Checking environment configuration...")
        
        env_file = Path('.env')
        if not env_file.exists():
            self.issues.append("❌ .env file not found")
            self.checks_failed += 1
            return
        
        # Check permissions (should not be world-readable)
        perms = oct(os.stat('.env').st_mode)[-3:]
        if perms[2] != '0':
            self.warnings.append("⚠️  .env file is world-readable (consider chmod 600)")
        
        # Check for required variables
        required_vars = [
            'SECRET_KEY', 'DB_PASSWORD', 'DB_NAME', 'DB_USER',
            'ALLOWED_HOSTS', 'DEBUG', 'DJANGO_ENV'
        ]
        
        env_content = env_file.read_text()
        missing_vars = []
        
        for var in required_vars:
            if f"{var}=" not in env_content:
                missing_vars.append(var)
        
        if missing_vars:
            self.issues.append(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            self.checks_failed += 1
        else:
            print("  ✓ All required environment variables present")
            self.checks_passed += 1
        
        # Check for default/weak values
        weak_patterns = [
            ('SECRET_KEY', 'django-insecure'),
            ('DB_PASSWORD', 'postgres'),
            ('DEBUG', 'True'),
            ('SECRET_KEY', 'change'),
            ('SECRET_KEY', 'your-secret'),
        ]
        
        for var, pattern in weak_patterns:
            if pattern.lower() in env_content.lower():
                self.warnings.append(f"⚠️  Possible weak/default value for {var}")
    
    def check_ssl_certificates(self):
        """Check if SSL certificates exist"""
        print("\n🔍 Checking SSL certificates...")
        
        ssl_dir = Path('nginx_ssl')
        if not ssl_dir.exists():
            self.warnings.append("⚠️  nginx_ssl directory not found")
            return
        
        cert_file = ssl_dir / 'fullchain.pem'
        key_file = ssl_dir / 'privkey.pem'
        
        if not cert_file.exists():
            self.issues.append("❌ SSL certificate (fullchain.pem) not found")
            self.checks_failed += 1
        else:
            print("  ✓ SSL certificate found")
            self.checks_passed += 1
        
        if not key_file.exists():
            self.issues.append("❌ SSL private key (privkey.pem) not found")
            self.checks_failed += 1
        else:
            print("  ✓ SSL private key found")
            self.checks_passed += 1
    
    def check_docker_files(self):
        """Check if required Docker files exist"""
        print("\n🔍 Checking Docker configuration...")
        
        required_files = [
            'Dockerfile.prod',
            'docker-compose.prod.yml',
            'entrypoint.prod.sh',
            'nginx.prod.conf',
        ]
        
        for file in required_files:
            if not Path(file).exists():
                self.issues.append(f"❌ Required file not found: {file}")
                self.checks_failed += 1
            else:
                print(f"  ✓ {file} found")
                self.checks_passed += 1
    
    def check_directory_structure(self):
        """Check if required directories exist"""
        print("\n🔍 Checking directory structure...")
        
        required_dirs = ['logs', 'db_backups', 'nginx_ssl']
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                self.warnings.append(f"⚠️  Directory not found: {dir_name} (will be created)")
                # Create the directory
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  📁 Created: {dir_name}")
            else:
                print(f"  ✓ {dir_name} directory exists")
                self.checks_passed += 1
    
    def check_gitignore(self):
        """Check if sensitive files are in .gitignore"""
        print("\n🔍 Checking .gitignore configuration...")
        
        gitignore = Path('.gitignore')
        if not gitignore.exists():
            self.warnings.append("⚠️  .gitignore not found")
            return
        
        content = gitignore.read_text()
        sensitive_patterns = ['.env', '*.pem', '*.key', 'db_backups', 'logs']
        
        missing = []
        for pattern in sensitive_patterns:
            if pattern not in content:
                missing.append(pattern)
        
        if missing:
            self.warnings.append(f"⚠️  Missing patterns in .gitignore: {', '.join(missing)}")
        else:
            print("  ✓ .gitignore properly configured")
            self.checks_passed += 1
    
    def check_python_compiled(self):
        """Check if build script exists"""
        print("\n🔍 Checking code compilation setup...")
        
        if Path('build_secure.py').exists():
            print("  ✓ Build script (build_secure.py) found")
            self.checks_passed += 1
        else:
            self.warnings.append("⚠️  build_secure.py not found")
    
    def check_security_headers(self):
        """Check if security headers are configured in nginx"""
        print("\n🔍 Checking security headers configuration...")
        
        nginx_conf = Path('nginx.prod.conf')
        if not nginx_conf.exists():
            return
        
        content = nginx_conf.read_text()
        security_headers = [
            'X-Frame-Options',
            'X-Content-Type-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
        ]
        
        missing = []
        for header in security_headers:
            if header not in content:
                missing.append(header)
        
        if missing:
            self.warnings.append(f"⚠️  Missing security headers: {', '.join(missing)}")
        else:
            print("  ✓ Security headers configured")
            self.checks_passed += 1
    
    def print_summary(self):
        """Print summary of security check"""
        print("\n" + "="*60)
        print("📊 SECURITY CHECK SUMMARY")
        print("="*60)
        
        print(f"\n✅ Checks passed: {self.checks_passed}")
        print(f"❌ Checks failed: {self.checks_failed}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.issues:
            print("\n🚨 CRITICAL ISSUES (Must fix before deployment):")
            for issue in self.issues:
                print(f"  {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS (Should address):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.issues and not self.warnings:
            print("\n🎉 All security checks passed! Ready for deployment.")
            print("\nNext steps:")
            print("  1. docker-compose -f docker-compose.prod.yml build")
            print("  2. docker-compose -f docker-compose.prod.yml up -d")
            print("  3. docker-compose -f docker-compose.prod.yml logs -f")
        elif not self.issues:
            print("\n✅ No critical issues found. Review warnings before deployment.")
        else:
            print("\n❌ Critical issues found. Please fix before deployment.")
            return 1
        
        print("\n" + "="*60)
        return 0


def main():
    """Main security check routine"""
    print("""
╔═══════════════════════════════════════════════════════╗
║     Maternal Backend - Security Check                ║
║     Validating deployment configuration              ║
╚═══════════════════════════════════════════════════════╝
    """)
    
    checker = SecurityChecker()
    
    # Run all checks
    checker.check_env_file()
    checker.check_ssl_certificates()
    checker.check_docker_files()
    checker.check_directory_structure()
    checker.check_gitignore()
    checker.check_python_compiled()
    checker.check_security_headers()
    
    # Print summary
    return checker.print_summary()


if __name__ == '__main__':
    sys.exit(main())

