FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 QUESTIONNAIRE_DATA_DIR=/data
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --create-home questionnaire && mkdir /data && chown questionnaire /data
COPY *.py /app/questionnaire/
COPY templates /app/questionnaire/templates/
COPY static /app/questionnaire/static/
USER questionnaire
EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 2 --timeout 120 --access-logfile - questionnaire.wsgi:app"]
