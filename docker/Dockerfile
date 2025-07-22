FROM python:3.9-slim

WORKDIR /app
COPY inference.py .
COPY model.py .
RUN pip install --upgrade pip && pip install torch transformers flask

EXPOSE 8080
ENV PORT=8080
CMD ["python", "inference.py"]
