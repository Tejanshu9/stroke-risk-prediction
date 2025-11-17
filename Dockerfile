# ===== Base Image =====
FROM python:3.13-slim   

# Install pipenv
RUN pip install pipenv


# Create working directory
WORKDIR /app

# Copy Pipenv files first to install dependencies
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies into system (no virtualenv)
RUN pipenv install --system --deploy

# Copy the prediction service and model
COPY ["predict.py", "stroke_model.bin", "./"]

# Expose API port
EXPOSE 9696

# Start the service with gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
