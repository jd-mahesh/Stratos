# Live-trader image, also on the Lambda base image.
#
# Build from the repo root:
#   docker build -f docker/live_trader.Dockerfile -t stratos-live-trader .
FROM public.ecr.aws/lambda/python:3.12

COPY requirements/ /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/live_trader.txt

COPY trader_core/ ${LAMBDA_TASK_ROOT}/trader_core/
COPY live_trader/ ${LAMBDA_TASK_ROOT}/live_trader/

# Lambda runs the code as a non-root user: make it readable (not writable) by everyone,
# whatever permissions the files had where the image was built.
RUN chmod -R a+rX ${LAMBDA_TASK_ROOT}/trader_core ${LAMBDA_TASK_ROOT}/live_trader

CMD ["live_trader.handler.handler"]
