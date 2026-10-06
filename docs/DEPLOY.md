# Deploying to AWS

This walks through putting Stratos on AWS by hand with the AWS CLI, the way the
author's own deployment was built. Doing it by hand once is the point: every command
maps to a box or arrow in the architecture diagram, and you'll understand what a tool
like Terraform is automating when you get to it later.

What you end up with:

```
EventBridge Scheduler ──every 5 min──▶ Lambda: live trader ──▶ Alpaca (paper)
                                            │        │
                                            │        └──▶ SNS ──▶ alert emails
                                            ▼
             Lambda: backtester ──▶ RDS Postgres ◀── dashboard (your laptop, or ECS)
                                            ▲
                               Secrets Manager (keys, DB password)
```

Everything below runs in a terminal on your own computer, in the repo root, with
Docker Desktop running. Rough time: 1.5 to 3 hours the first time, most of it waiting
on RDS and reading error messages.

**Treat it like a real account.** Stratos only paper trades, but every step here is
written as if real money were at stake: no secret is ever printed, typed into a
command line or written to a file, each AWS role gets only the permissions it needs,
and nothing trades until you've watched a day of dry runs.

---

## 0. Before touching anything

1. **Set a budget alert first.** Billing console → Budgets → Create budget → "Zero spend
   budget", plus a monthly cost budget of about $10. You'll get an email the moment
   anything costs money (on the free plan, that includes spending credits).
2. **Never use the root user for this.** Newer AWS accounts (the "projects" sign-up) have
   no root login and give you a role to work as; older accounts should create an IAM
   Identity Center user. Either way you end up with short-lived credentials, not
   long-lived access keys.
3. **Install the AWS CLI v2** (the official installer). If your shell says
   `command not found: aws` afterwards, open a new terminal or add `~/.local/bin` to `PATH`.
4. **Log in.** Recent CLI versions have `aws login`, which opens your browser and keeps
   you signed in for 12 hours; older setups use `aws configure sso`. Use a named profile:

   ```bash
   aws login --profile stratos
   aws sts get-caller-identity --profile stratos   # shows your account and role
   ```

5. **Pick your region.** New accounts may be assigned one (the author's got `us-east-2`,
   Ohio). Any region works; just use the same one everywhere.

Set these in every new terminal you use for AWS work:

```bash
export AWS_PROFILE=stratos AWS_REGION=us-east-2 AWS_PAGER=""   # AWS_PAGER="" stops the CLI opening a pager
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
```

---

## 1. Database: RDS Postgres

### The networking decision (read this, it's the most expensive gotcha in the project)

RDS normally lives in a private network (VPC). If you put the Lambda functions in that
VPC so they can reach a *private* database, they lose internet access, and the live
trader needs the internet to reach Alpaca. The standard fix is a **NAT gateway**, which
costs roughly $30+/month just for existing, more than everything else here combined.

For a learning project we take the simpler route: a **publicly reachable** database
protected by a long random password and forced SSL, with Lambda staying outside the VPC.
It's a real trade-off (anyone can *try* to connect to the database port), and a good one
to be able to explain in an interview. Before trading real money, move to private
subnets with a NAT gateway or VPC endpoints.

```bash
# A dedicated firewall (security group) for the database
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
DB_SG=$(aws ec2 create-security-group --group-name stratos-db \
  --description "Postgres for Stratos" --vpc-id $VPC_ID --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $DB_SG --protocol tcp --port 5432 --cidr 0.0.0.0/0

# A random password, kept in a shell variable only. It's never printed; step 2 stores it
# in Secrets Manager, which becomes the only place it lives.
DB_PASSWORD=$(openssl rand -hex 20)

aws rds create-db-instance \
  --db-instance-identifier stratos-db \
  --engine postgres \
  --db-instance-class db.t4g.micro \
  --allocated-storage 20 --storage-type gp3 --storage-encrypted \
  --master-username trader \
  --master-user-password "$DB_PASSWORD" \
  --db-name trading \
  --vpc-security-group-ids $DB_SG \
  --publicly-accessible \
  --backup-retention-period 1 \
  --deletion-protection \
  --no-multi-az --query 'DBInstance.DBInstanceStatus'

# Takes 5–15 minutes; the prompt comes back when it's ready
aws rds wait db-instance-available --db-instance-identifier stratos-db
DB_HOST=$(aws rds describe-db-instances --db-instance-identifier stratos-db \
  --query 'DBInstances[0].Endpoint.Address' --output text)
DATABASE_URL="postgresql+psycopg2://trader:$DB_PASSWORD@$DB_HOST:5432/trading?sslmode=require"
```

Check it from your laptop by running a backtest straight into RDS. `--no-deps` stops
Compose from starting the local Postgres, and `-e DATABASE_URL` with no value passes the
variable through without putting the password on the command line:

```bash
docker compose --profile jobs build backtester
DATABASE_URL="$DATABASE_URL" docker compose run --rm --no-deps -e DATABASE_URL backtester --provider synthetic
```

It should end with "Saved as run 1".

---

## 2. Secrets Manager

One secret holds everything sensitive. The code reads it when `SECRET_ID` is set
(see `trader_core/config.py`), so no key is ever baked into an image or a Lambda setting.

This builds the secret from your `.env` (the Alpaca paper keys) and the database URL,
adds a random dashboard password, and pipes it straight to AWS: nothing is printed,
written to disk or put on a command line where other programs could see it.

```bash
DATABASE_URL="$DATABASE_URL" python3 - <<'EOF' | \
  aws secretsmanager create-secret --name stratos/prod --secret-string file:///dev/stdin --query ARN --output text
import json, os, secrets
from trader_core.config import read_dotenv
env = read_dotenv()
print(json.dumps({
    "ALPACA_API_KEY_ID": env["ALPACA_API_KEY_ID"],
    "ALPACA_API_SECRET_KEY": env["ALPACA_API_SECRET_KEY"],
    "DATABASE_URL": os.environ["DATABASE_URL"],
    "DASHBOARD_PASSWORD": secrets.token_urlsafe(24),
}))
EOF
SECRET_ARN=$(aws secretsmanager describe-secret --secret-id stratos/prod --query ARN --output text)
unset DB_PASSWORD   # it's in Secrets Manager now
```

If you ever regenerate your Alpaca keys, update both `.env` and this secret.

---

## 3. ECR repositories

```bash
for r in backtester live-trader dashboard; do
  aws ecr create-repository --repository-name stratos-$r \
    --image-scanning-configuration scanOnPush=true >/dev/null
  # keep only the 5 newest images so storage doesn't pile up
  aws ecr put-lifecycle-policy --repository-name stratos-$r --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep 5","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}' >/dev/null
done
```

## 4. Build and push the images

```bash
bash scripts/push_images.sh
```

The script logs Docker in to ECR, builds each image for the right CPU architecture,
tags it with the current git commit, pushes it, and points any existing Lambda function
at the new version. It uses `AWS_REGION` (and falls back to `us-east-1` if that isn't
set, so set it). Read the comments in it: the `--provenance=false` flag and the
arm64/amd64 split are both things that trip people up. The Dockerfiles also make the
code readable by the non-root users the containers run as (`chmod -R a+rX`), since files
copied from a Mac can be owner-only.

A large layer (the dashboard's is about 500 MB) can fail with "use of closed network
connection"; just run the same command again.

---

## 5. IAM role for the Lambda functions

A role is the identity a function runs as. This one may write logs and read one secret,
nothing else. (Step 9 adds sending alerts to one topic.)

```bash
aws iam create-role --role-name stratos-lambda --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy --role-name stratos-lambda \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam put-role-policy --role-name stratos-lambda --policy-name read-trading-secret \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"$SECRET_ARN\"}]}"

LAMBDA_ROLE=$(aws iam get-role --role-name stratos-lambda --query Role.Arn --output text)
```

---

## 6. Lambda functions

The functions get their strategy settings from your `.env`, minus anything secret. This
builds that list (keys, passwords and `DATABASE_URL` are left out on purpose):

```bash
lambda_env() {  # usage: lambda_env KEY=VALUE ...   (extra settings to add or override)
  python3 - "$@" <<'EOF'
import json, sys
from trader_core.config import read_dotenv
env = read_dotenv()
keep = ["SYMBOLS", "STRATEGY", "MOMENTUM_LOOKBACK", "MOMENTUM_TOP", "MOMENTUM_VOL_SCALE", "SIGNAL_MODE",
        "FRACTIONAL_SHARES", "CRASH_SWITCH", "FAST_WINDOW", "SLOW_WINDOW", "TREND_WINDOW"]
out = {k: env[k] for k in keep if env.get(k)}
out["SECRET_ID"] = "stratos/prod"
out.update(dict(arg.split("=", 1) for arg in sys.argv[1:]))
print(json.dumps({"Variables": out}))
EOF
}
```

If `create-function` says the role can't be assumed, IAM hasn't propagated yet. Wait 10 seconds and retry.

```bash
aws lambda create-function --function-name stratos-backtester \
  --package-type Image --code ImageUri=$REGISTRY/stratos-backtester:latest \
  --role $LAMBDA_ROLE --architectures arm64 --timeout 300 --memory-size 1024 \
  --environment "$(lambda_env DATA_PROVIDER=alpaca)" --query State

# DRY_RUN=true to start: it decides and records everything but sends no orders.
aws lambda create-function --function-name stratos-live-trader \
  --package-type Image --code ImageUri=$REGISTRY/stratos-live-trader:latest \
  --role $LAMBDA_ROLE --architectures arm64 --timeout 120 --memory-size 1024 \
  --environment "$(lambda_env DRY_RUN=true)" --query State

aws lambda wait function-active-v2 --function-name stratos-live-trader
```

A live-trader run takes 3 to 6 seconds and about 260 MB when it rebalances, and
0.2 seconds on the "nothing to do this month" runs.

Run a backtest in the cloud:

```bash
aws lambda invoke --function-name stratos-backtester \
  --cli-binary-format raw-in-base64-out --cli-read-timeout 310 \
  --payload '{"start": "2021-01-01"}' /tmp/out.json && head -c 600 /tmp/out.json
```

Poke the live trader (outside market hours it answers `"market_closed"`):

```bash
aws lambda invoke --function-name stratos-live-trader /tmp/out.json && cat /tmp/out.json
```

### Changing one setting safely

`update-function-configuration --environment` **replaces every variable**, so leaving one
out deletes it. This pattern reads the current settings, changes exactly one, and writes
them back:

```bash
set_lambda_setting() {  # usage: set_lambda_setting KEY VALUE
  local env_json
  env_json=$(aws lambda get-function-configuration --function-name stratos-live-trader --query Environment --output json \
    | KEY="$1" VALUE="$2" python3 -c 'import os,sys,json; e=json.load(sys.stdin); e["Variables"][os.environ["KEY"]]=os.environ["VALUE"]; print(json.dumps(e))')
  aws lambda update-function-configuration --function-name stratos-live-trader \
    --environment "$env_json" --query "Environment.Variables.$1"
  aws lambda wait function-updated --function-name stratos-live-trader
}
```

When you're happy with a day of dry-run decisions (step 8 shows how to read them):

```bash
set_lambda_setting DRY_RUN false
```

The same helper sets anything else, for example `set_lambda_setting CAPITAL_RESERVE 100000`
to trade only what's above $100k (README, "Trading a small budget"). A new setting takes
effect on the next run; that run's log shows "Init Duration" because Lambda starts fresh.
Changing a strategy setting or the reserve makes the next run rebalance right away.

---

## 7. EventBridge Scheduler: every 5 minutes, weekdays

The schedule is written in New York time, so it follows daylight saving automatically.
It fires from 9:00 to 15:55; the function checks Alpaca's market clock and exits early
before the 9:30 open and on market holidays.

```bash
aws iam create-role --role-name stratos-scheduler --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

LIVE_ARN=$(aws lambda get-function --function-name stratos-live-trader --query Configuration.FunctionArn --output text)
aws iam put-role-policy --role-name stratos-scheduler --policy-name invoke-live-trader \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"$LIVE_ARN\"}]}"
SCHED_ROLE=$(aws iam get-role --role-name stratos-scheduler --query Role.Arn --output text)

# The Input passes the scheduled time into the function; a retried invocation gets the
# same value, so its orders reuse the same client_order_id and Alpaca rejects the repeat.
aws scheduler create-schedule --name stratos-every-5-min \
  --schedule-expression "cron(0/5 9-15 ? * MON-FRI *)" \
  --schedule-expression-timezone "America/New_York" \
  --flexible-time-window '{"Mode":"OFF"}' \
  --target "{\"Arn\":\"$LIVE_ARN\",\"RoleArn\":\"$SCHED_ROLE\",\"Input\":\"{\\\"scheduled_time\\\": \\\"<aws.scheduler.scheduled-time>\\\"}\"}"
```

To pause trading, disable the schedule in the EventBridge Scheduler console (one
toggle), or set `TRADING_HALTED=true` with `set_lambda_setting`, which keeps recording
balances but places no orders.

---

## 8. Watching it run

Once the schedule is on, everything is automatic: your computer can be off. To see what
it decided, read the logs (pick a `--since` window that reaches back far enough; a
rebalance you're looking for may be older than you think):

```bash
# the latest run's decisions for the stocks you hold, plus any trades or problems
aws logs tail /aws/lambda/stratos-live-trader --since 30m --format short \
  | grep -wE "buy|sell|error|ERROR|WARNING|Traceback|halted|Init|done|REPORT"

# any errors, timeouts or safety warnings in the last 4 hours, across every run
aws logs filter-log-events --log-group-name /aws/lambda/stratos-live-trader \
  --start-time $(( ($(date +%s) - 4*3600) * 1000 )) \
  --filter-pattern '?ERROR ?Traceback ?"Task timed out" ?WARNING' --query 'events[].message' --output text
```

After the month's rebalance, each run logs one line,
`momentum rebalances monthly; done for 2026-10, next in November 2026`.

To check the function's settings and whether an update has finished:

```bash
aws lambda get-function-configuration --function-name stratos-live-trader \
  --query '{dry_run:Environment.Variables.DRY_RUN,reserve:Environment.Variables.CAPITAL_RESERVE,status:LastUpdateStatus,modified:LastModified}'
```

**Shipping a code change:** run the tests, commit, then
`bash scripts/push_images.sh live-trader` and `aws lambda wait function-updated --function-name stratos-live-trader`.
Editing code on your laptop changes nothing in the cloud until you push. Never run the
live trader for real (without `--dry-run`) on your laptop while the cloud one is trading
the same account.

---

## 9. Alert emails (SNS)

When a safeguard trips (a circuit-breaker halt, a cancelled run, a failed order, bad
price data, unsafe settings), the live trader emails you through an SNS topic, at most
once a day per problem. Without this, those only show up in the logs.

```bash
TOPIC_ARN=$(aws sns create-topic --name stratos-alerts --query TopicArn --output text)
aws sns subscribe --topic-arn "$TOPIC_ARN" --protocol email --notification-endpoint you@example.com
```

Click **Confirm subscription** in the email from AWS Notifications (check spam). Then let
the function send to that one topic and tell it where the topic is:

```bash
aws iam put-role-policy --role-name stratos-lambda --policy-name publish-alerts \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"sns:Publish\",\"Resource\":\"$TOPIC_ARN\"}]}"
set_lambda_setting ALERT_TOPIC_ARN "$TOPIC_ARN"
```

Test it end to end, through the function's own permissions:

```bash
aws lambda invoke --function-name stratos-live-trader --cli-binary-format raw-in-base64-out \
  --payload '{"test_alert": true}' /tmp/out.json && cat /tmp/out.json
```

`"status": "alert_sent"` plus a "Stratos: test alert" email means it works; otherwise
the response says why. It never trades. Trust this test over IAM's policy simulator:
on accounts created through the newer sign-up, `aws iam simulate-principal-policy` can
report `explicitDeny` from an organization rule for a publish that actually works.

### Status emails: online, trades, offline

Problem alerts are always on once the topic is set. To also hear about normal operation,
turn on status emails:

```bash
set_lambda_setting STATUS_EMAILS true
```

Each trading day you then get "Stratos is online" from the first run after the open, a
"Stratos bought …" or "Stratos sold …" email from any run that places orders, and "Stratos is
offline for the day" (with the day's summary) from the last run before the close, sent as
that run's final step. Early-close days are handled automatically. Dry runs never send them.

### Watchdog: an alert if Stratos stops running

Stratos can't report its own outage, so a second schedule checks on it every 15 minutes
during market hours. If no run has finished in 15 minutes while the market is open, it emails
"Stratos alert: offline" once per outage. It reuses the scheduler role and never trades.

```bash
aws scheduler create-schedule --name stratos-watchdog \
  --schedule-expression "cron(0/15 9-15 ? * MON-FRI *)" \
  --schedule-expression-timezone "America/New_York" \
  --flexible-time-window '{"Mode":"OFF"}' \
  --target "{\"Arn\":\"$LIVE_ARN\",\"RoleArn\":\"$SCHED_ROLE\",\"Input\":\"{\\\"watchdog\\\": true}\"}"
```

(Before 9:45 it stays quiet, since the day's first runs may not have happened yet.)

### Crash alarm

If runs start failing outright, a CloudWatch alarm emails you within about 5 minutes:

```bash
aws cloudwatch put-metric-alarm --alarm-name stratos-live-trader-errors \
  --namespace AWS/Lambda --metric-name Errors \
  --dimensions Name=FunctionName,Value=stratos-live-trader \
  --statistic Sum --period 300 --evaluation-periods 1 --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold --treat-missing-data notBreaching \
  --alarm-actions "$TOPIC_ARN"

# test it: force the alarm on (you should get an email), then it resets itself on the next check
aws cloudwatch set-alarm-state --alarm-name stratos-live-trader-errors \
  --state-value ALARM --state-reason "Testing the Stratos crash alarm"
```

SNS email is free at this volume (the first 1,000 emails a month), and the first 10
CloudWatch alarms are free.

---

## 10. Dashboard

### On your laptop (free)

```bash
docker compose build dashboard
bash scripts/dashboard.sh
# then open http://localhost:8502 (Ctrl+C to stop)
```

The script reads `DATABASE_URL` from Secrets Manager without printing it and runs the
dashboard image on `127.0.0.1` only, so other devices on your network can't reach it.
It shows the same data the cloud bot writes. If a browser shows a blank page or a
"RangeError" popup, something else (another Streamlit app) is on that port or the
browser cached an old page: stop the other app, or use a private window.

### On AWS: ECS Express Mode (optional, about $40–45/month)

**Why not App Runner?** AWS closed App Runner to new customers on April 30, 2026 and now
points people to **Amazon ECS Express Mode**, which from one container image creates an
ECS service on Fargate, an Application Load Balancer, auto scaling, networking and an
HTTPS URL. It's also the more useful thing to have on a résumé.

**Cost warning.** The load balancer and the Fargate task bill by the hour even when nobody
is looking, which makes this the most expensive part of the project. Deploy it when you
need a public link (say, for applications) and delete it afterwards.

```bash
# The role the task runs with (pulls the image, reads the secret) ...
aws iam create-role --role-name stratos-dashboard-exec --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name stratos-dashboard-exec \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name stratos-dashboard-exec --policy-name read-trading-secret \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"$SECRET_ARN\"}]}"

# ... and the role ECS uses to build the load balancer and networking for you
aws iam create-role --role-name stratos-ecs-infra --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name stratos-ecs-infra \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices

EXEC_ROLE=$(aws iam get-role --role-name stratos-dashboard-exec --query Role.Arn --output text)
INFRA_ROLE=$(aws iam get-role --role-name stratos-ecs-infra --query Role.Arn --output text)

bash scripts/push_images.sh dashboard

aws ecs create-express-gateway-service --service-name stratos-dashboard \
  --execution-role-arn $EXEC_ROLE --infrastructure-role-arn $INFRA_ROLE \
  --primary-container "{\"image\":\"$REGISTRY/stratos-dashboard:latest\",\"containerPort\":8080,\"secrets\":[{\"name\":\"DATABASE_URL\",\"valueFrom\":\"$SECRET_ARN:DATABASE_URL::\"},{\"name\":\"DASHBOARD_PASSWORD\",\"valueFrom\":\"$SECRET_ARN:DASHBOARD_PASSWORD::\"}]}" \
  --health-check-path "/_stcore/health" --cpu 0.5 --memory 1 \
  --scaling-target '{"minTaskCount":1,"maxTaskCount":1}' --monitor-resources
```

Notes: the `:KEY::` suffix pulls one field out of the JSON secret. The health check path
must be `/_stcore/health` (Streamlit's), or the load balancer keeps killing the task.
When it's healthy, the ECS console shows the service's HTTPS URL. The dashboard asks for
`DASHBOARD_PASSWORD`; copy it to your clipboard without displaying it:

```bash
aws secretsmanager get-secret-value --secret-id stratos/prod --query SecretString --output text \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["DASHBOARD_PASSWORD"], end="")' | pbcopy
```

---

## 11. Costs and tearing down

What costs money (on the free plan, it comes out of your credits):

| Piece | About per month |
|---|---|
| RDS db.t4g.micro running 24/7 | $12 |
| RDS storage (20 GB) | $2.30 |
| Public IPv4 address for the database | $3.65 |
| Secrets Manager (1 secret) | $0.40 |
| ECR image storage | under $0.50 |
| Lambda, Scheduler, CloudWatch logs, SNS email | about $0 (inside their free allowances) |
| **Without the ECS dashboard** | **about $18–19** |
| ECS Express dashboard (Fargate + load balancer + IPs) | add about $40–45 |

New AWS accounts get a credit-based free plan (the author's had $200 in credits for 6
months). The plan closes the account when the credits run out or the 6 months end,
unless you upgrade to a paid plan. Check **Billing** and your budget alerts rather than
trusting any number here.

To stop the database for a while without deleting it:
`aws rds stop-db-instance --db-instance-identifier stratos-db`
(AWS restarts stopped instances automatically after 7 days).

Delete everything:

```bash
aws scheduler delete-schedule --name stratos-every-5-min
aws scheduler delete-schedule --name stratos-watchdog
aws cloudwatch delete-alarms --alarm-names stratos-live-trader-errors
aws lambda delete-function --function-name stratos-live-trader
aws lambda delete-function --function-name stratos-backtester
# ECS Express dashboard, if you deployed it: delete the service in the ECS console (this also removes its load balancer)
aws rds modify-db-instance --db-instance-identifier stratos-db --no-deletion-protection --apply-immediately >/dev/null
aws rds delete-db-instance --db-instance-identifier stratos-db --skip-final-snapshot >/dev/null
aws rds wait db-instance-deleted --db-instance-identifier stratos-db
aws ec2 delete-security-group --group-id $DB_SG
aws secretsmanager delete-secret --secret-id stratos/prod --force-delete-without-recovery
aws sns delete-topic --topic-arn "$TOPIC_ARN"
for r in backtester live-trader dashboard; do aws ecr delete-repository --repository-name stratos-$r --force; done
aws iam delete-role-policy --role-name stratos-scheduler --policy-name invoke-live-trader
aws iam delete-role --role-name stratos-scheduler
aws iam delete-role-policy --role-name stratos-lambda --policy-name read-trading-secret
aws iam delete-role-policy --role-name stratos-lambda --policy-name publish-alerts
aws iam detach-role-policy --role-name stratos-lambda --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name stratos-lambda
```

If you created the dashboard roles, delete those too (detach their policies first, the
same way).
