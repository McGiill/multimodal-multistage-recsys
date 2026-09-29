FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements/ requirements/
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/api.txt
COPY src/ src/
COPY configs/ configs/
EXPOSE 8000
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
