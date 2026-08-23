FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY validator.py validate.py dimer_transport.py dimer_entrypoint.py ./
# Baked model-specific fallback for DIMER Custom / Other pipelines that do not
# resolve a task type generically. The validator's semantics are already
# regression-specific; this preserves the fallback at the platform layer.
ENV DIMER_TASK_TYPE=tabular_regression
# The wrapper preserves ordinary local/PVC execution and stages an on-prem
# S3_DATASET_PREFIX only when DIMER_DATASET_DIR has no local files.
CMD ["python", "dimer_entrypoint.py"]
