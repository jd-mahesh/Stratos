# Dashboard image: a normal long-running web container (not Lambda).
#
# Build from the repo root:
#   docker build -f docker/dashboard.Dockerfile -t stratos-dashboard .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements/ requirements/
RUN pip install -r requirements/dashboard.txt

COPY trader_core/ trader_core/
COPY dashboard/ dashboard/

# Make the code readable (not writable) by the non-root user below, whatever
# permissions the files had where the image was built.
RUN chmod -R a+rX /app/trader_core /app/dashboard

# Don't run as root inside the container.
RUN useradd --create-home app
USER app

EXPOSE 8080
# The load balancer checks /_stcore/health, Streamlit's built-in health endpoint.
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8080", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
