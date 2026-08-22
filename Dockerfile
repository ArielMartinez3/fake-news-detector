FROM python:3.11-slim

WORKDIR /app

# System dependencies for spaCy and matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Application code
COPY . .

# Train the model during build (optional — remove if you prefer runtime training)
# RUN python train.py

# Expose Gradio default port
EXPOSE 7860

# Launch the interactive demo
CMD ["python", "app/app.py"]
