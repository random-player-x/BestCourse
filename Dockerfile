# Use an official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


COPY app/ app/
COPY main.py .

# Copy dependency files
COPY requirements.txt .

# ENV HUGGINGFACE_HUB_TOKEN='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'

RUN huggingface-cli login --token $HUGGINGFACE_HUB_TOKEN





# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .



# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
