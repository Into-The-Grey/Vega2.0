#!/bin/bash
# Vega 2.0 Cleanup Script
# ======================
#
# This script performs safe cleanup operations on the Vega2.0 codebase:
# 1. Removes __pycache__ directories
# 2. Cleans SQLite temporary files (.db-shm, .db-wal)
# 3. Archives old log files (optional)
#
# Usage:
#     bash scripts/cleanup.sh [--full]
#     
#     --full: Includes log archiving (default: skip logs)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/home/ncacord/Vega2.0"

echo -e "${GREEN}=== Vega 2.0 Cleanup Script ===${NC}\n"

# Function to print with color
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from project root
if [ ! -f "$PROJECT_ROOT/main.py" ]; then
    print_error "Please run this script from the Vega2.0 project root"
    exit 1
fi

# Step 1: Remove __pycache__ directories
print_info "Cleaning __pycache__ directories..."
pycache_count=$(find "$PROJECT_ROOT" -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ "$pycache_count" -gt 0 ]; then
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    print_info "Removed $pycache_count __pycache__ directories"
else
    print_info "No __pycache__ directories found"
fi

# Step 2: Clean SQLite temporary files (only if Vega is not running)
print_info "Checking for running Vega processes..."
if pgrep -f "vega.*main.py.*server" > /dev/null; then
    print_warning "Vega server is running. Skipping SQLite temp file cleanup."
    print_warning "Stop Vega with: pkill -f 'vega.*main.py.*server'"
else
    print_info "Cleaning SQLite temporary files (.db-shm, .db-wal)..."
    shm_count=$(find "$PROJECT_ROOT" -type f -name "*.db-shm" 2>/dev/null | wc -l)
    wal_count=$(find "$PROJECT_ROOT" -type f -name "*.db-wal" 2>/dev/null | wc -l)
    
    if [ "$shm_count" -gt 0 ] || [ "$wal_count" -gt 0 ]; then
        find "$PROJECT_ROOT" -type f \( -name "*.db-shm" -o -name "*.db-wal" \) -delete 2>/dev/null || true
        print_info "Removed $shm_count .db-shm files and $wal_count .db-wal files"
    else
        print_info "No SQLite temporary files found"
    fi
fi

# Step 3: Archive old logs (if --full flag is passed)
if [ "$1" == "--full" ]; then
    print_info "Archiving old log files (older than 30 days)..."
    
    # Create archive directory
    mkdir -p "$PROJECT_ROOT/logs/archive"
    
    # Find and move old logs
    old_logs=$(find "$PROJECT_ROOT/logs" -type f -name "*.log" -mtime +30 2>/dev/null | wc -l)
    
    if [ "$old_logs" -gt 0 ]; then
        find "$PROJECT_ROOT/logs" -maxdepth 1 -type f -name "*.log" -mtime +30 -exec mv {} "$PROJECT_ROOT/logs/archive/" \; 2>/dev/null || true
        
        # Compress archived logs
        archive_date=$(date +%Y%m%d)
        if [ -n "$(ls -A "$PROJECT_ROOT/logs/archive"/*.log 2>/dev/null)" ]; then
            tar -czf "$PROJECT_ROOT/logs/archive_${archive_date}.tar.gz" -C "$PROJECT_ROOT/logs/archive" . 2>/dev/null || true
            rm -f "$PROJECT_ROOT/logs/archive"/*.log 2>/dev/null || true
            print_info "Archived $old_logs log files to archive_${archive_date}.tar.gz"
        fi
    else
        print_info "No old log files to archive"
    fi
else
    print_info "Skipping log archiving (use --full flag to enable)"
fi

# Step 4: Summary
echo ""
print_info "Cleanup complete!"
echo ""
print_info "Summary:"
echo "  - __pycache__ directories removed: $pycache_count"
if pgrep -f "vega.*main.py.*server" > /dev/null; then
    echo "  - SQLite temp files: SKIPPED (Vega is running)"
else
    echo "  - SQLite temp files removed: $((shm_count + wal_count))"
fi

if [ "$1" == "--full" ]; then
    echo "  - Log files archived: $old_logs"
fi

echo ""
print_info "Disk space freed:"
du -sh "$PROJECT_ROOT" | awk '{print "  Total project size: " $1}'

# Optional: Show git status
echo ""
print_info "Git status:"
cd "$PROJECT_ROOT"
if git rev-parse --git-dir > /dev/null 2>&1; then
    git status --short | head -10
    echo ""
    print_info "Run 'git add .' and 'git commit' to save changes"
else
    print_warning "Not a git repository"
fi

echo ""
print_info "Done! 🎉"
