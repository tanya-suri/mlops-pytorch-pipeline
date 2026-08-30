# MLOps PyTorch Pipeline on Kubernetes

An end-to-end Machine Learning pipeline for training and serving PyTorch vision models using Kubernetes (Minikube).

---

## **Architecture Overview**

```text
[ Local Environment ]
        │
        ▼
[ Kubernetes PVC: mlops-pvc ]
        │
        ├───────────────────────────────┐
        ▼                               ▼
[ Training Job: pytorch-training ]   [ Serving Deployment: model-serving ]
  - Trains ResNet Model                - 2 Replicas
  - Saves checkpoint to PVC             - Mounts PVC (/app/checkpoints)
                                       - Health Probes (/health)
                                                │
                                                ▼
                                     [ Service: model-serving ]
                                                │
                                                ▼
                                     [ REST API: /predict ]




content = """# Repository Structure

```text
mlops-pytorch-pipeline/
├── configs/
│   └── training_config.json      # Training hyperparameters & config
├── docker/
│   ├── Dockerfile.train          # Image for PyTorch training job
│   └── Dockerfile.serve          # Image for FastAPI serving container
├── k8s/
│   ├── namespace.yaml            # ml-training namespace
│   ├── pvc.yaml                  # PersistentVolumeClaim definition
│   ├── configmap.yaml            # ConfigMap for hyperparameters
│   ├── training-job.yaml         # Kubernetes Batch Job for training
│   ├── serving-deployment.yaml   # Deployment specs (2 replicas, probes, resources)
│   └── serving-service.yaml      # ClusterIP service for model serving
├── src/
│   ├── train.py                  # Model training logic
│   └── serve.py                  # FastAPI server with health & prediction endpoints
├── tests/
│   └── test_api.py               # API unit & integration tests
├── requirements.txt              # Dependency specifications
└── README.md                     # Setup & architecture guide





content = """# Quick Start Setup

## 1. Environment Prerequisites
Ensure you have Docker, Minikube, and kubectl installed on your machine.

```bash
# Start Minikube cluster
minikube start

# Point terminal to Minikube's Docker daemon
eval $(minikube -p minikube docker-env)



content = """## 2. Build Container Images
Build the Docker images directly inside Minikube's environment:

```bash
# Build training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Build serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .





content = """## 3. Deploy Cluster Infrastructure
Apply the namespace, persistent storage, and training configuration maps:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml


content = """## 4. Execute PyTorch Training Job
Trigger the Kubernetes batch job to train the model and output `classifier_v1.pt` to the shared PVC:

```bash
kubectl apply -f k8s/training-job.yaml

# Watch training pod status until Completed
kubectl get pods -n ml-training -w



5. Deploy Model Serving Layer
Once training completes, start the serving deployment and service:
Bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml

# Verify deployment health
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training


content = """## 6. Port-Forward and Test REST Endpoints

### Port Forwarding
Open port forwarding in your primary terminal:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training



content = """### Health Check (`GET /health`)
In a secondary terminal window:

```bash
curl -i http://localhost:8080/health
