# Telecommunication

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Telecommunication Commission Prediction System

A machine learning system that predicts commission values for telecommunication products using historical data. This project implements a complete ML pipeline with data versioning, experiment tracking, workflow orchestration, and containerized deployment.

## 🌟 Features

- Automated ML pipeline for commission prediction
- Real-time predictions via FastAPI endpoint
- Model versioning and experiment tracking with MLflow
- Data versioning with DVC
- Workflow orchestration using Prefect
- Containerized deployment with Docker
- Standardized project structure using Cookiecutter

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/telco-commission-prediction.git
cd telco-commission-prediction
```

2. Initialize project structure using Cookiecutter:
```bash
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

3. Initialize DVC and add data directories:
```bash
# Initialize DVC
dvc init

# Add data and models directories to DVC
dvc add data/
dvc add models/

# Create remote storage (replace 'your-remote-storage' with actual storage path)
dvc remote add -d storage your-remote-storage

# Push to remote storage
dvc push
```

4. Start the Docker services (API and MLflow):
```bash
docker-compose up -d
```

5. Access the API and MLflow services:
- FastAPI application: http://localhost:8000
- MLflow UI: http://localhost:5000

## 🔄 Workflow Orchestration with Prefect (Local Setup)

1. Open a new terminal and start the Prefect server:
```bash
prefect server start
```

2. Open another terminal and start the Prefect agent:
```bash
prefect agent start -p default-agent-pool
```

3. Deploy and run the pipeline:
```bash
python -m telecommunication.pipeline.deployment
```

4. Access the Prefect UI to monitor workflows:
```bash
http://localhost:4200
```

The Prefect UI allows you to:
- Monitor pipeline runs
- View task execution status
- Check logs and error messages
- Manage workflow deployments
