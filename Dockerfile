# Docker image for Render: same Python as Colab, so model.joblib loads with sklearn 1.6.1
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py model.joblib ./
COPY data/ data/
COPY artifacts/ artifacts/
# Render passes the port in $PORT (7860 if it is not set, e.g. running Docker locally)
RUN useradd -m -u 1000 user
USER user
EXPOSE 7860
CMD streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0
