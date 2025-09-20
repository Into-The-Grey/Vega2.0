# Kubernetes Deployment Guide for Vega 2.0

This directory contains Kubernetes manifests for deploying Vega 2.0 in a production environment.

## Quick Start

```bash
# Deploy everything
kubectl apply -f k8s/

# Or deploy step by step
kubectl apply -f k8s/vega-api.yaml
kubectl apply -f k8s/vega-federated.yaml
kubectl apply -f k8s/monitoring.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/deployment-utils.yaml
```

## Files Overview

- **vega-api.yaml**: Main API server deployment with persistent storage
- **vega-federated.yaml**: Federated learning coordinator with model storage
- **monitoring.yaml**: Prometheus monitoring and Redis cache
- **ingress.yaml**: Nginx ingress with TLS termination and network policies
- **deployment-utils.yaml**: Deployment scripts and RBAC configurations

## Configuration

### Secrets

Update the secrets in `vega-api.yaml` with base64-encoded values:

```bash
# Generate API key secret
echo -n "your-actual-api-key" | base64

# Update the secret in vega-api.yaml
```

### Storage

Adjust PVC sizes based on your needs:

- **vega-data**: 10Gi (database and app data)
- **vega-models**: 50Gi (federated learning models)
- **redis-data**: 5Gi (cache)
- **prometheus-data**: 20Gi (metrics retention)

### Ingress

Update domain names in `ingress.yaml`:

```yaml
rules:
- host: your-domain.com  # Update this
```

## Monitoring

Prometheus scrapes metrics from:

- Vega API server (`:8000/metrics`)
- Federated coordinator (`:8001/metrics`)
- Redis (`:6379`)

Access Prometheus UI:

```bash
kubectl port-forward svc/prometheus-service 9090:9090 -n vega-system
```

## Scaling

Scale the API server:

```bash
kubectl scale deployment/vega-api -n vega-system --replicas=5
```

Or use the provided script:

```bash
kubectl exec -n vega-system deployment/vega-api -- bash /scripts/scale.sh 5
```

## Federated Learning

The federated coordinator runs as a single replica and manages:

- Participant registration
- Model aggregation
- Cross-silo federation
- Multi-task learning coordination

## Security Features

- Network policies restrict traffic
- Service accounts with minimal RBAC permissions
- Non-root containers with dropped capabilities
- TLS termination at ingress
- Secret management for sensitive data

## Troubleshooting

Check pod status:

```bash
kubectl get pods -n vega-system
kubectl describe pod <pod-name> -n vega-system
kubectl logs <pod-name> -n vega-system
```

Check services:

```bash
kubectl get services -n vega-system
kubectl describe service vega-api-service -n vega-system
```

Check persistent volumes:

```bash
kubectl get pvc -n vega-system
kubectl describe pvc vega-data -n vega-system
```

## Backup Strategy

Regular backups of persistent data:

```bash
# Database backup
kubectl exec -n vega-system deployment/vega-api -- sqlite3 /app/data/vega.db ".backup /app/data/backup-$(date +%Y%m%d).db"

# Model backup
kubectl exec -n vega-system deployment/vega-federated-coordinator -- tar -czf /app/models/backup-$(date +%Y%m%d).tar.gz /app/models/
```

## Resource Requirements

Minimum cluster requirements:

- **CPU**: 4 vCPUs total
- **Memory**: 8GB RAM total
- **Storage**: 100GB persistent storage

Recommended for production:

- **CPU**: 8+ vCPUs
- **Memory**: 16GB+ RAM
- **Storage**: 500GB+ persistent storage
- **Nodes**: 3+ for high availability
