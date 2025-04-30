# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app

# Expose port 5000 for the Flask application
EXPOSE 8080

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Install gunicorn
RUN pip install gunicorn

# Set the entrypoint to run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app", "--workers", "1", "--timeout", "60"]