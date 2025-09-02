# MLOps Architecture

This document describes the architecture of the MLOps project, including data flow, training pipeline, serving path, and MLflow interactions.

## Overview

The MLOps project implements a complete machine learning pipeline from data ingestion to model serving, with MLflow as the central experiment tracking and model registry system.

## Architecture Components

```mermaid
graph TB
    subgraph "Data Layer"
        CSV[CSV Files]
        VAL[Data Validation]
        FE[Feature Engineering]
        SPLIT[Train/Val/Test Split]
    end
    
    subgraph "Training Layer"
        TRAIN[Model Training]
        CV[Cross Validation]
        HPO[Hyperparameter Optimization]
        EVAL[Model Evaluation]
    end
    
    subgraph "MLflow Layer"
        TRACK[Experiment Tracking]
        REG[Model Registry]
        ART[Artifact Storage]
    end
    
    subgraph "Serving Layer"
        API[FastAPI Server]
        MODEL[Production Model]
        METRICS[Prometheus Metrics]
        HEALTH[Health Checks]
    end
    
    subgraph "Infrastructure"
        K8S[Kubernetes]
        DOCKER[Docker]
        CI[CI/CD Pipeline]
    end
    
    CSV --> VAL
    VAL --> FE
    FE --> SPLIT
    SPLIT --> TRAIN
    TRAIN --> CV
    CV --> HPO
    HPO --> EVAL
    EVAL --> TRACK
    TRACK --> REG
    REG --> ART
    REG --> MODEL
    MODEL --> API
    API --> METRICS
    API --> HEALTH
    TRAIN --> DOCKER
    API --> DOCKER
    DOCKER --> K8S
    CI --> DOCKER
```

## Data Flow

### 1. Data Ingestion and Validation

```mermaid
flowchart LR
    A[CSV Files] --> B[DataLoader]
    B --> C[Schema Detection]
    C --> D[Data Validation]
    D --> E[Quality Metrics]
    E --> F[MLflow Logging]
    
    subgraph "Validation Steps"
        G[Null Check]
        H[Type Check]
        I[Leakage Detection]
        J[Statistical Analysis]
    end
    
    D --> G
    D --> H
    D --> I
    D --> J
    G --> F
    H --> F
    I --> F
    J --> F
```

### 2. Feature Engineering

```mermaid
flowchart LR
    A[Raw Data] --> B[ColumnTransformer]
    B --> C[Numeric Features]
    B --> D[Categorical Features]
    
    C --> E[Imputation]
    E --> F[Scaling]
    
    D --> G[Imputation]
    G --> H[Encoding]
    
    F --> I[Feature Selection]
    H --> I
    I --> J[Fitted Transformer]
    J --> K[MLflow Logging]
```

### 3. Model Training Pipeline

```mermaid
flowchart TD
    A[Training Data] --> B[Model Candidates]
    B --> C[Logistic Regression]
    B --> D[Random Forest]
    B --> E[XGBoost]
    B --> F[LightGBM]
    B --> G[PyTorch MLP]
    
    C --> H[Cross Validation]
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I[Optuna HPO]
    I --> J[Best Model Selection]
    J --> K[Model Persistence]
    K --> L[MLflow Logging]
    
    subgraph "Deep Learning Features"
        M[Automatic Mixed Precision]
        N[Early Stopping]
        O[Learning Rate Scheduling]
        P[Gradient Clipping]
        Q[Class Weights]
    end
    
    G --> M
    G --> N
    G --> O
    G --> P
    G --> Q
```

### 4. Model Evaluation

```mermaid
flowchart LR
    A[Test Data] --> B[Model Prediction]
    B --> C[Metrics Calculation]
    
    C --> D[Classification Metrics]
    C --> E[Regression Metrics]
    
    D --> F[Accuracy, Precision, Recall, F1]
    D --> G[ROC AUC, PR AUC]
    D --> H[Confusion Matrix]
    
    E --> I[MSE, RMSE, MAE, R²]
    E --> J[Residual Plots]
    
    F --> K[SHAP Analysis]
    G --> K
    H --> K
    I --> K
    J --> K
    
    K --> L[Model Card Generation]
    L --> M[MLflow Logging]
```

### 5. Model Registry and Promotion

```mermaid
flowchart TD
    A[Evaluated Model] --> B[MLflow Model Registry]
    B --> C[Model Registration]
    C --> D[Promotion Gates]
    
    D --> E{F1 ≥ 0.8?}
    D --> F{Accuracy ≥ 0.85?}
    D --> G{ROC AUC ≥ 0.9?}
    
    E --> H[Pass/Fail]
    F --> H
    G --> H
    
    H --> I{All Gates Pass?}
    I -->|Yes| J[Promote to Staging]
    I -->|No| K[Keep in None Stage]
    
    J --> L{Manual Review}
    L -->|Approve| M[Promote to Production]
    L -->|Reject| N[Demote to None]
    
    M --> O[Archive Previous Production]
```

### 6. Model Serving

```mermaid
flowchart LR
    A[Production Model] --> B[FastAPI Server]
    B --> C[/healthz]
    B --> D[/predict]
    B --> E[/metrics]
    B --> F[/model-info]
    
    C --> G[Health Status]
    D --> H[Prediction Service]
    E --> I[Prometheus Metrics]
    F --> J[Model Metadata]
    
    H --> K[Input Validation]
    K --> L[Feature Preprocessing]
    L --> M[Model Inference]
    M --> N[Response Formatting]
    
    subgraph "Monitoring"
        O[Request Counters]
        P[Response Times]
        Q[Error Rates]
        R[Model Performance]
    end
    
    H --> O
    H --> P
    H --> Q
    M --> R
```

## Kubernetes Deployment

### 1. Training Jobs

```mermaid
graph TB
    subgraph "Training Namespace"
        TJ[Training Job]
        PVC1[Data PVC]
        PVC2[Artifacts PVC]
        PVC3[MLflow PVC]
    end
    
    subgraph "MLflow Namespace"
        MS[MLflow Server]
        MSVC[MLflow Service]
    end
    
    TJ --> PVC1
    TJ --> PVC2
    TJ --> PVC3
    TJ --> MSVC
    MSVC --> MS
```

### 2. Serving Deployment

```mermaid
graph TB
    subgraph "Serving Namespace"
        DEP[Deployment]
        SVC[Service]
        HPA[Horizontal Pod Autoscaler]
        ING[Ingress]
    end
    
    subgraph "MLflow Namespace"
        MS[MLflow Server]
        MSVC[MLflow Service]
    end
    
    DEP --> SVC
    SVC --> ING
    DEP --> HPA
    DEP --> MSVC
    MSVC --> MS
```

## MLflow Integration

### 1. Experiment Tracking

```mermaid
flowchart LR
    A[Training Run] --> B[MLflow Tracking]
    B --> C[Parameters]
    B --> D[Metrics]
    B --> E[Artifacts]
    
    C --> F[Model Config]
    C --> G[Hyperparameters]
    
    D --> H[Training Metrics]
    D --> I[Validation Metrics]
    D --> J[Test Metrics]
    
    E --> K[Model Files]
    E --> L[Feature Engineer]
    E --> M[Evaluation Reports]
    E --> N[SHAP Plots]
```

### 2. Model Registry

```mermaid
flowchart TD
    A[Model Logging] --> B[Model Registry]
    B --> C[Model Versioning]
    C --> D[Stage Management]
    
    D --> E[None]
    D --> F[Staging]
    D --> G[Production]
    D --> H[Archived]
    
    E --> I[Development]
    F --> J[Testing]
    G --> K[Serving]
    H --> L[Historical]
    
    K --> M[Model Serving]
    M --> N[Performance Monitoring]
    N --> O[Model Updates]
    O --> P[New Version]
    P --> A
```

## CI/CD Pipeline

```mermaid
flowchart TD
    A[Code Push] --> B[CI Pipeline]
    B --> C[Code Quality]
    B --> D[Unit Tests]
    B --> E[Integration Tests]
    
    C --> F[Ruff Linting]
    C --> G[Black Formatting]
    C --> H[MyPy Type Checking]
    
    D --> I[Pytest Coverage]
    E --> J[End-to-End Tests]
    
    F --> K{All Checks Pass?}
    G --> K
    H --> K
    I --> K
    J --> K
    
    K -->|Yes| L[Build Docker Images]
    K -->|No| M[Fail Build]
    
    L --> N[Security Scan]
    N --> O[Push to Registry]
    O --> P[Deploy to K8s]
```

## Security and Monitoring

### 1. Security Features

- Non-root Docker containers
- Kubernetes RBAC
- Secrets management
- Network policies
- Pod security standards

### 2. Monitoring Stack

```mermaid
graph TB
    subgraph "Application Metrics"
        AM[FastAPI Metrics]
        PM[Prometheus]
        GM[Grafana]
    end
    
    subgraph "Infrastructure Metrics"
        IM[Kubernetes Metrics]
        NM[Node Exporter]
        CM[Container Runtime]
    end
    
    subgraph "MLflow Monitoring"
        MM[MLflow Metrics]
        LM[Log Aggregation]
        AM[Alert Manager]
    end
    
    AM --> PM
    IM --> PM
    MM --> PM
    PM --> GM
    PM --> AM
```

## Data Persistence

### 1. Storage Classes

- **Data PVC**: Training datasets and validation data
- **Artifacts PVC**: Model artifacts and evaluation results
- **MLflow PVC**: Experiment tracking and model registry data

### 2. Backup Strategy

- Automated daily backups
- Point-in-time recovery
- Cross-region replication (optional)
- Retention policies

## Scalability Considerations

### 1. Horizontal Scaling

- HPA for serving pods
- Multiple training job replicas
- Load balancing across instances

### 2. Vertical Scaling

- Resource requests and limits
- GPU scheduling
- Memory optimization
- CPU allocation

### 3. Storage Scaling

- Dynamic volume provisioning
- Storage class selection
- Backup and archival policies

## Disaster Recovery

### 1. Backup Procedures

- Database backups
- Artifact backups
- Configuration backups
- Documentation backups

### 2. Recovery Procedures

- Service restoration
- Data restoration
- Configuration restoration
- Validation procedures

## Performance Optimization

### 1. Training Optimization

- GPU utilization
- Memory management
- Batch size optimization
- Distributed training support

### 2. Serving Optimization

- Model caching
- Request batching
- Async processing
- Connection pooling

### 3. Infrastructure Optimization

- Resource allocation
- Network optimization
- Storage performance
- Monitoring overhead
