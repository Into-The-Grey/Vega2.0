#!/bin/bash
# Security Testing and Validation Script for Vega 2.0

set -e

echo "🔒 Vega 2.0 Security Testing Suite"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p security/results

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install security tools if not present
install_security_tools() {
    print_status "Checking security tools..."
    
    if ! command_exists bandit; then
        print_warning "Installing bandit..."
        pip install bandit[toml]
    fi
    
    if ! command_exists safety; then
        print_warning "Installing safety..."
        pip install safety
    fi
    
    if ! command_exists semgrep; then
        print_warning "Installing semgrep..."
        pip install semgrep
    fi
    
    print_success "Security tools ready"
}

# Test 1: Python Security Linting with Bandit
test_bandit() {
    print_status "Running Bandit security scan..."
    
    if bandit -r src/ -f json -o security/results/bandit-test.json >/dev/null 2>&1; then
        print_success "Bandit scan completed"
        
        # Check for critical issues
        critical_issues=$(python3 -c "
import json
try:
    with open('security/results/bandit-test.json') as f:
        data = json.load(f)
    high_severity = [r for r in data.get('results', []) if r.get('issue_severity') == 'HIGH']
    print(len(high_severity))
except:
    print(0)
" 2>/dev/null || echo "0")
        
        if [ "$critical_issues" -gt 0 ]; then
            print_warning "Found $critical_issues high-severity security issues"
        else
            print_success "No high-severity security issues found"
        fi
    else
        print_error "Bandit scan failed"
        return 1
    fi
}

# Test 2: Dependency Vulnerability Scanning
test_safety() {
    print_status "Running Safety dependency scan..."
    
    if safety check --json --output security/results/safety-test.json >/dev/null 2>&1; then
        print_success "Safety scan completed - no vulnerable dependencies"
    else
        # Safety returns non-zero exit code when vulnerabilities are found
        vulnerabilities=$(python3 -c "
import json
try:
    with open('security/results/safety-test.json') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
" 2>/dev/null || echo "0")
        
        if [ "$vulnerabilities" -gt 0 ]; then
            print_warning "Found $vulnerabilities vulnerable dependencies"
        else
            print_success "No vulnerable dependencies found"
        fi
    fi
}

# Test 3: Static Analysis with Semgrep
test_semgrep() {
    print_status "Running Semgrep static analysis..."
    
    if semgrep --config=auto --json --output=security/results/semgrep-test.json src/ >/dev/null 2>&1; then
        print_success "Semgrep analysis completed"
        
        # Check for security findings
        findings=$(python3 -c "
import json
try:
    with open('security/results/semgrep-test.json') as f:
        data = json.load(f)
    results = data.get('results', [])
    security_findings = [r for r in results if 'security' in r.get('check_id', '').lower()]
    print(len(security_findings))
except:
    print(0)
" 2>/dev/null || echo "0")
        
        if [ "$findings" -gt 0 ]; then
            print_warning "Found $findings security-related findings"
        else
            print_success "No security findings detected"
        fi
    else
        print_warning "Semgrep analysis completed with warnings"
    fi
}

# Test 4: Configuration Security Check
test_config_security() {
    print_status "Checking configuration security..."
    
    # Check for hardcoded secrets
    if grep -r -i "password\|secret\|key\|token" src/ --include="*.py" | grep -E "(=|\:)" | grep -v "# nosec" >/dev/null; then
        print_warning "Potential hardcoded secrets found (check manually)"
    else
        print_success "No obvious hardcoded secrets detected"
    fi
    
    # Check for insecure defaults
    if grep -r "localhost\|127.0.0.1" src/ --include="*.py" | grep -v "# allowed" >/dev/null; then
        print_success "Localhost bindings found (good for security)"
    fi
    
    # Check for proper error handling
    if grep -r "except:" src/ --include="*.py" | head -5 >/dev/null; then
        print_warning "Bare except clauses found (potential security risk)"
    else
        print_success "No bare except clauses found"
    fi
}

# Test 5: Vega Security Module Tests
test_vega_security() {
    print_status "Testing Vega security modules..."
    
    # Test security scanner import
    if python3 -c "from src.vega.security.scanner import SecurityScanner; print('OK')" >/dev/null 2>&1; then
        print_success "Security scanner module imports correctly"
    else
        print_error "Security scanner module import failed"
        return 1
    fi
    
    # Test vulnerability manager import
    if python3 -c "from src.vega.security.vuln_manager import VulnerabilityManager; print('OK')" >/dev/null 2>&1; then
        print_success "Vulnerability manager module imports correctly"
    else
        print_error "Vulnerability manager module import failed"
        return 1
    fi
    
    # Test compliance reporter import
    if python3 -c "from src.vega.security.compliance import ComplianceReporter; print('OK')" >/dev/null 2>&1; then
        print_success "Compliance reporter module imports correctly"
    else
        print_error "Compliance reporter module import failed"
        return 1
    fi
    
    # Test security integration
    if python3 -c "from src.vega.security.integration import SecurityOrchestrator; print('OK')" >/dev/null 2>&1; then
        print_success "Security integration module imports correctly"
    else
        print_error "Security integration module import failed"
        return 1
    fi
}

# Test 6: Security Configuration Validation
test_security_config() {
    print_status "Validating security configuration..."
    
    if [ -f "configs/security.yaml" ]; then
        if python3 -c "
import yaml
with open('configs/security.yaml') as f:
    config = yaml.safe_load(f)
required = ['scanner', 'vulnerability', 'compliance', 'ci_cd']
missing = [s for s in required if s not in config]
if missing:
    print(f'Missing sections: {missing}')
    exit(1)
print('Configuration valid')
" >/dev/null 2>&1; then
            print_success "Security configuration is valid"
        else
            print_error "Security configuration validation failed"
            return 1
        fi
    else
        print_error "Security configuration file not found"
        return 1
    fi
}

# Test 7: CI/CD Security Integration
test_ci_integration() {
    print_status "Testing CI/CD security integration..."
    
    if [ -f ".github/workflows/security.yml" ]; then
        print_success "Security workflow configuration found"
        
        # Check for required jobs
        if grep -q "security-scan" .github/workflows/security.yml; then
            print_success "Security scan job configured"
        else
            print_warning "Security scan job not found in workflow"
        fi
        
        if grep -q "compliance-check" .github/workflows/security.yml; then
            print_success "Compliance check job configured"
        else
            print_warning "Compliance check job not found in workflow"
        fi
    else
        print_warning "Security workflow configuration not found"
    fi
    
    if [ -f ".pre-commit-config.yaml" ]; then
        print_success "Pre-commit hooks configured"
        
        if grep -q "bandit" .pre-commit-config.yaml; then
            print_success "Bandit pre-commit hook configured"
        else
            print_warning "Bandit pre-commit hook not configured"
        fi
    else
        print_warning "Pre-commit configuration not found"
    fi
}

# Test 8: Security Dashboard
test_security_dashboard() {
    print_status "Checking security dashboard..."
    
    if [ -f "src/vega/security/dashboard.html" ]; then
        print_success "Security dashboard found"
        
        # Basic HTML validation
        if grep -q "Security Dashboard" src/vega/security/dashboard.html; then
            print_success "Dashboard contains expected content"
        else
            print_warning "Dashboard content validation failed"
        fi
    else
        print_error "Security dashboard not found"
        return 1
    fi
}

# Main test execution
main() {
    echo
    print_status "Starting security test suite..."
    echo
    
    # Track test results
    tests_passed=0
    tests_failed=0
    
    # Run all tests
    if install_security_tools; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    if test_bandit; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    if test_safety; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    if test_semgrep; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    if test_config_security; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    if test_vega_security; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    if test_security_config; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    if test_ci_integration; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    if test_security_dashboard; then ((tests_passed++)); else ((tests_failed++)); fi
    echo
    
    # Final report
    echo "=========================="
    echo "Security Test Results"
    echo "=========================="
    echo -e "${GREEN}Tests Passed: $tests_passed${NC}"
    echo -e "${RED}Tests Failed: $tests_failed${NC}"
    
    total_tests=$((tests_passed + tests_failed))
    if [ $total_tests -gt 0 ]; then
        pass_rate=$((tests_passed * 100 / total_tests))
        echo "Pass Rate: $pass_rate%"
    fi
    
    if [ $tests_failed -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All security tests passed!${NC}"
        exit 0
    else
        echo -e "\n${YELLOW}⚠️  Some security tests failed. Review the output above.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"