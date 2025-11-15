#!/usr/bin/env python3
"""
Secure Build Script for Maternal Backend
Compiles Python files to .pyc and optionally creates PyInstaller executable
"""
import os
import py_compile
import shutil
import sys
from pathlib import Path


class SecureBuilder:
    def __init__(self, source_dir='.', build_dir='build_secure'):
        self.source_dir = Path(source_dir).resolve()
        self.build_dir = Path(build_dir).resolve()
        self.exclude_dirs = {
            'venv', 'env', '__pycache__', 'node_modules', 
            '.git', 'build', 'dist', 'staticfiles', 'media',
            'build_secure', 'builds', 'artifacts'
        }
        self.exclude_files = {'.pyc', '.pyo', '.pyd', '.so', '.dll'}
        
    def clean_build_dir(self):
        """Remove old build directory"""
        if self.build_dir.exists():
            print(f"Cleaning old build directory: {self.build_dir}")
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
    def should_process(self, path):
        """Check if file/directory should be processed"""
        path_parts = Path(path).parts
        return not any(excluded in path_parts for excluded in self.exclude_dirs)
    
    def compile_python_file(self, src_file, dest_dir):
        """Compile a single Python file to .pyc"""
        try:
            # Create destination directory
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Compile to .pyc
            compiled = py_compile.compile(
                src_file, 
                cfile=dest_dir / f"{src_file.stem}.pyc",
                doraise=True
            )
            print(f"✓ Compiled: {src_file.relative_to(self.source_dir)}")
            return True
        except Exception as e:
            print(f"✗ Error compiling {src_file}: {e}")
            return False
    
    def copy_non_python_file(self, src_file, dest_file):
        """Copy non-Python files (config, data, etc.)"""
        try:
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            return True
        except Exception as e:
            print(f"✗ Error copying {src_file}: {e}")
            return False
    
    def build_compiled_version(self):
        """Build compiled version with .pyc files"""
        print("\n=== Building Secure Compiled Version ===\n")
        self.clean_build_dir()
        
        stats = {'compiled': 0, 'copied': 0, 'errors': 0}
        
        # Walk through source directory
        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)
            
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            if not self.should_process(root_path):
                continue
            
            # Calculate relative path
            try:
                rel_path = root_path.relative_to(self.source_dir)
            except ValueError:
                continue
                
            dest_dir = self.build_dir / rel_path
            
            for file in files:
                src_file = root_path / file
                
                if file.endswith('.py'):
                    # Compile Python files
                    if self.compile_python_file(src_file, dest_dir):
                        stats['compiled'] += 1
                    else:
                        stats['errors'] += 1
                        
                elif not any(file.endswith(ext) for ext in self.exclude_files):
                    # Copy other necessary files
                    dest_file = dest_dir / file
                    if self.copy_non_python_file(src_file, dest_file):
                        stats['copied'] += 1
                        print(f"→ Copied: {file}")
        
        # Copy essential root files
        essential_files = [
            'manage.py', 'requirements.txt', 'nginx.conf',
            'docker-compose.yml', 'docker-compose.prod.yml',
            'Dockerfile.prod', 'entrypoint.prod.sh', '.env.example'
        ]
        
        for filename in essential_files:
            src = self.source_dir / filename
            if src.exists():
                dest = self.build_dir / filename
                if self.copy_non_python_file(src, dest):
                    stats['copied'] += 1
        
        # Create __init__.py files for Python packages
        self.create_init_files()
        
        print(f"\n=== Build Complete ===")
        print(f"✓ Compiled: {stats['compiled']} Python files")
        print(f"→ Copied: {stats['copied']} other files")
        print(f"✗ Errors: {stats['errors']}")
        print(f"\nOutput directory: {self.build_dir}")
        
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        for root, dirs, files in os.walk(self.build_dir):
            # If directory contains .pyc files, ensure it has __init__.pyc
            pyc_files = [f for f in files if f.endswith('.pyc')]
            if pyc_files:
                init_file = Path(root) / '__init__.pyc'
                if not init_file.exists():
                    # Create empty __init__.py and compile it
                    temp_init = Path(root) / '__init__.py'
                    temp_init.write_text('')
                    py_compile.compile(temp_init, cfile=init_file)
                    temp_init.unlink()
                    print(f"+ Created: {init_file.relative_to(self.build_dir)}")
    
    def create_deployment_package(self):
        """Create a deployment package"""
        print("\n=== Creating Deployment Package ===\n")
        
        # Create tar.gz archive
        archive_name = 'maternal_backend_secure'
        shutil.make_archive(
            self.source_dir / archive_name,
            'gztar',
            self.build_dir
        )
        
        print(f"✓ Created: {archive_name}.tar.gz")


def main():
    """Main build process"""
    print("""
╔═══════════════════════════════════════════════════════╗
║     Maternal Backend - Secure Build System           ║
║     Compiling Python files for production            ║
╚═══════════════════════════════════════════════════════╝
    """)
    
    builder = SecureBuilder()
    
    # Build compiled version
    builder.build_compiled_version()
    
    # Optionally create deployment package
    create_package = input("\nCreate deployment package (.tar.gz)? [y/N]: ")
    if create_package.lower() == 'y':
        builder.create_deployment_package()
    
    print("\n✓ Build process complete!")
    print("\nNext steps:")
    print("1. Review the build_secure/ directory")
    print("2. Test the compiled version")
    print("3. Build Docker image: docker-compose -f docker-compose.prod.yml build")
    print("4. Deploy: docker-compose -f docker-compose.prod.yml up -d")


if __name__ == '__main__':
    main()

