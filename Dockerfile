FROM python:3.12

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the needed checkpoint files
COPY checkpoints/sft_transformer.pt ./checkpoints/
COPY checkpoints/sft_final.pt    ./checkpoints/

# Copy tokenizer + code
COPY data/tokenizer*.json ./data/
COPY *.py ./

EXPOSE 8502

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]
