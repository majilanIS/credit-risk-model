## Task 6 – Model Deployment & Continuous Integration

**Files:**

- `src/api/main.py`
- `src/api/pydantic_models.py`
- `Dockerfile`
- `docker-compose.yml`
- `.github/workflows/ci.yml`

**Objective:**  
Package the trained model into a containerized FastAPI service and set up a CI/CD pipeline to automate testing and enforce code quality.

---

### Steps

1. **API Development**

   - Built a REST API using **FastAPI** in `src/api/main.py`.
   - The API loads the best model from the **MLflow registry** (Production stage).
   - Implemented a `/predict` endpoint that accepts new customer transaction data (validated with **Pydantic models**) and returns the risk probability.
   - Added a `/health` endpoint for service monitoring.

2. **Containerization**

   - Created a `Dockerfile` to build a lightweight Python 3.12 image with FastAPI and Uvicorn.
   - Configured `docker-compose.yml` to simplify service startup and environment variable management (e.g., `MODEL_NAME`, `MLFLOW_TRACKING_URI`).

3. **CI/CD Pipeline**
   - Added a GitHub Actions workflow in `.github/workflows/ci.yml`.
   - Pipeline runs automatically on every push to the `main` branch.
   - Steps include:
     - **Linting** with `flake8` to enforce code style.
     - **Testing** with `pytest` to validate functionality.
   - Build fails if either linting or tests fail, ensuring code quality before merging.

---

### Run Commands

```bash
# Build and run locally with Docker Compose
docker compose up --build

# Access API docs
http://localhost:8000/docs
```
