FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/simulator.txt
COPY src/ src/
COPY configs/ configs/
CMD ["python", "-m", "src.simulator.stream"]
