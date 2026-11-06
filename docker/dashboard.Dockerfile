FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements/ requirements/
RUN pip install --no-cache-dir --root-user-action=ignore -c requirements/constraints.txt -r requirements/dashboard.txt
COPY dashboard/ dashboard/
COPY configs/ configs/
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
