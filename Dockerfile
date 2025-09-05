# Use official Streamlit image (smaller, optimized)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy dependencies first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit default port
EXPOSE 8080

# Run Streamlit app
CMD ["streamlit", "run", "cvd_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
