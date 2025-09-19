FROM python:3.11-slim

WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Now copy the rest of the application code
COPY . .

# The command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--threads", "8", "--timeout", "0", "main:app"]