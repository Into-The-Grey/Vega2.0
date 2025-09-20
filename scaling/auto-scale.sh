#!/bin/bash

# Vega 2.0 Auto-scaling Configuration Script
# This script sets up auto-scaling for various components based on load

set -e

# Configuration
API_MIN_REPLICAS=${API_MIN_REPLICAS:-2}
API_MAX_REPLICAS=${API_MAX_REPLICAS:-10}
FEDERATED_MIN_REPLICAS=${FEDERATED_MIN_REPLICAS:-1}
FEDERATED_MAX_REPLICAS=${FEDERATED_MAX_REPLICAS:-3}

# Namespace
NAMESPACE=${NAMESPACE:-vega-system}

echo "🚀 Configuring Vega 2.0 Auto-scaling..."

# Function to check if resource exists
resource_exists() {
    kubectl get "$1" "$2" -n "$NAMESPACE" >/dev/null 2>&1
}

# Function to apply scaling configuration
apply_scaling() {
    local component=$1
    local min_replicas=$2
    local max_replicas=$3
    
    echo "📊 Configuring scaling for $component (min: $min_replicas, max: $max_replicas)"
    
    # Update HPA if it exists
    if resource_exists hpa "${component}-hpa"; then
        kubectl patch hpa "${component}-hpa" -n "$NAMESPACE" --type='merge' -p="{\"spec\":{\"minReplicas\":$min_replicas,\"maxReplicas\":$max_replicas}}"
        echo "✅ Updated HPA for $component"
    else
        echo "⚠️  HPA not found for $component, applying from file"
        kubectl apply -f scaling/hpa.yaml -n "$NAMESPACE"
    fi
}

# Function to get current metrics
get_current_metrics() {
    echo "📈 Current System Metrics:"
    
    # API metrics
    if resource_exists deployment vega-api; then
        local api_replicas=$(kubectl get deployment vega-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        local api_ready=$(kubectl get deployment vega-api -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        echo "   API Server: $api_ready/$api_replicas replicas ready"
        
        # Get CPU/Memory usage if metrics-server is available
        if kubectl top nodes >/dev/null 2>&1; then
            kubectl top pods -n "$NAMESPACE" -l app=vega-api | tail -n +2 | while read line; do
                echo "   $line"
            done
        fi
    fi
    
    # Federated coordinator metrics
    if resource_exists deployment vega-federated-coordinator; then
        local fed_replicas=$(kubectl get deployment vega-federated-coordinator -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        local fed_ready=$(kubectl get deployment vega-federated-coordinator -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        echo "   Federated Coordinator: $fed_ready/$fed_replicas replicas ready"
    fi
}

# Function to configure load balancing
configure_load_balancing() {
    echo "⚖️  Configuring load balancing..."
    
    # Update service to use session affinity for stateful operations
    kubectl patch service vega-api-service -n "$NAMESPACE" --type='merge' -p='{"spec":{"sessionAffinity":"ClientIP","sessionAffinityConfig":{"clientIP":{"timeoutSeconds":3600}}}}'
    
    # Configure federated coordinator for high availability
    if resource_exists deployment vega-federated-coordinator; then
        kubectl patch deployment vega-federated-coordinator -n "$NAMESPACE" --type='merge' -p='{"spec":{"strategy":{"type":"RollingUpdate","rollingUpdate":{"maxSurge":1,"maxUnavailable":0}}}}'
    fi
    
    echo "✅ Load balancing configured"
}

# Function to set up scaling policies
setup_scaling_policies() {
    echo "📋 Setting up scaling policies..."
    
    # Apply HPA configurations
    kubectl apply -f scaling/hpa.yaml -n "$NAMESPACE"
    
    # Apply VPA configurations if VPA is installed
    if kubectl get crd verticalpodautoscalers.autoscaling.k8s.io >/dev/null 2>&1; then
        kubectl apply -f scaling/vpa.yaml -n "$NAMESPACE"
        echo "✅ VPA policies applied"
    else
        echo "⚠️  VPA not available, skipping vertical scaling configuration"
    fi
    
    echo "✅ Scaling policies configured"
}

# Function to test scaling
test_scaling() {
    echo "🧪 Testing scaling configuration..."
    
    # Generate some load (if curl is available)
    if command -v curl >/dev/null 2>&1; then
        echo "   Generating test load..."
        for i in {1..10}; do
            kubectl run test-load-$i --image=curlimages/curl --rm -it --restart=Never -- \
                sh -c "for j in {1..50}; do curl -s http://vega-api-service:8000/healthz >/dev/null; sleep 0.1; done" &
        done
        
        echo "   Load generation started, check scaling in 2-3 minutes"
        echo "   Monitor with: kubectl get hpa -n $NAMESPACE"
    else
        echo "   curl not available, skipping load test"
    fi
}

# Function to monitor scaling
monitor_scaling() {
    echo "👀 Monitoring scaling status..."
    
    echo "HPA Status:"
    kubectl get hpa -n "$NAMESPACE"
    
    echo -e "\nDeployment Status:"
    kubectl get deployments -n "$NAMESPACE" -o wide
    
    echo -e "\nPod Resource Usage (if available):"
    if kubectl top nodes >/dev/null 2>&1; then
        kubectl top pods -n "$NAMESPACE"
    else
        echo "   Metrics server not available"
    fi
}

# Function to cleanup test resources
cleanup_test() {
    echo "🧹 Cleaning up test resources..."
    kubectl delete pods -n "$NAMESPACE" -l run=test-load --force --grace-period=0 >/dev/null 2>&1 || true
    echo "✅ Cleanup complete"
}

# Main execution
case "${1:-setup}" in
    "setup")
        get_current_metrics
        apply_scaling "vega-api" "$API_MIN_REPLICAS" "$API_MAX_REPLICAS"
        apply_scaling "vega-federated-coordinator" "$FEDERATED_MIN_REPLICAS" "$FEDERATED_MAX_REPLICAS"
        configure_load_balancing
        setup_scaling_policies
        echo "🎉 Auto-scaling configuration complete!"
        ;;
    "test")
        test_scaling
        ;;
    "monitor")
        monitor_scaling
        ;;
    "cleanup")
        cleanup_test
        ;;
    "status")
        get_current_metrics
        monitor_scaling
        ;;
    *)
        echo "Usage: $0 {setup|test|monitor|cleanup|status}"
        echo "  setup   - Configure auto-scaling (default)"
        echo "  test    - Generate test load"
        echo "  monitor - Show current scaling status"
        echo "  cleanup - Clean up test resources"
        echo "  status  - Show current metrics and status"
        exit 1
        ;;
esac