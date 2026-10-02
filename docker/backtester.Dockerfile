# Backtester image. Built on AWS's Lambda base image, which contains the Lambda
# runtime plus a local emulator, so the same image runs on Lambda and on your laptop.
#
# Build from the repo root:
#   docker build -f docker/backtester.Dockerfile -t stratos-backtester .
FROM public.ecr.aws/lambda/python:3.12

COPY requirements/ /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/backtester.txt

COPY trader_core/ ${LAMBDA_TASK_ROOT}/trader_core/
COPY backtester/ ${LAMBDA_TASK_ROOT}/backtester/

# Lambda runs the code as a non-root user: make it readable (not writable) by everyone,
# whatever permissions the files had where the image was built.
RUN chmod -R a+rX ${LAMBDA_TASK_ROOT}/trader_core ${LAMBDA_TASK_ROOT}/backtester

# Lambda calls this function. docker compose overrides the entrypoint to run the CLI instead.
CMD ["backtester.handler.handler"]
