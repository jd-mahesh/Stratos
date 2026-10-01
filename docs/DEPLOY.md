# Deploying to AWS

This walks through putting the three containers on AWS by hand with the AWS CLI.
Doing it by hand once is the point: every command maps to a box or arrow in the
architecture diagram, and you'll understand what a tool like Terraform is
automating when you get to it later.

Everything below assumes region `us-east-1` and a shell on your Mac in the repo root.

Rough time: 1.5 to 3 hours the first time, most of it waiting on RDS and reading error messages.

---

## 0. Before touching anything

1. **Set a budget alert first.** Billing console → Budgets → Create budget → "Zero spend budget"
   or a monthly cost budget of about $10. You'll get an email the moment anything costs money.
2. **Don't use the root user.** Create an IAM user (or IAM Identity Center user) with
   `AdministratorAccess` for this learning project and use its access keys.
3. Install the **AWS CLI v2** and run `aws configure` (region `us-east-1`, output `json`).
   Check it works: `aws sts get-caller-identity`.
4. **Docker Desktop** running.

Set a few shell variables you'll reuse. Re-run this block if you open a new terminal.

```bash
export AWS_REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
```

---

## 1. Database: RDS Postgres

### The networking decision (read this, it's the most expensive gotcha in the project)

RDS normally lives in a private network (VPC). If you put the Lambda functions in that
VPC so they can reach a *private* database, they lose internet access, and the live
trader needs the internet to reach Alpaca. The standard fix is a **NAT gateway**, which
costs roughly $30+/month just for existing. That's more than everything else here combined.

For a learning project we take the simpler route: a **publicly reachable** database
protected by a long random password and forced SSL, with Lambda staying outside the VPC.
It's a real trade-off (the database port is open to the internet), and it's a good
thing to be able to explain in an interview. The production-grade version is private
subnets + NAT or VPC endpoints.

```bash
# A dedicated firewall (security group) for the database
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
DB_SG=$(aws ec2 create-security-group --group-name stratos-db \
  --description "Postgres for Stratos" --vpc-id $VPC_ID --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $DB_SG --protocol tcp --port 5432 --cidr 0.0.0.0/0

# Letters and digits only, so it can go in a URL without escaping. Save it somewhere safe.
DB_PASSWORD=$(openssl rand -hex 20)
echo "$DB_PASSWORD"

aws rds create-db-instance \
  --db-instance-identifier stratos-db \
  --engine postgres \
  --db-instance-class db.t4g.micro \
  --allocated-storage 20 \
  --master-username trader \
  --master-user-password "$DB_PASSWORD" \
  --db-name trading \
  --vpc-security-group-ids $DB_SG \
  --publicly-accessible \
  --backup-retention-period 1 \
  --no-multi-az

# Takes 5–15 minutes
aws rds wait db-instance-available --db-instance-identifier stratos-db
DB_HOST=$(aws rds describe-db-instances --db-instance-identifier stratos-db \
  --query 'DBInstances[0].Endpoint.Address' --output text)

DATABASE_URL="postgresql+psycopg2://trader:$DB_PASSWORD@$DB_HOST:5432/trading?sslmode=require"
echo "$DATABASE_URL"
```

Check it from your laptop by running a backtest straight into RDS:

```bash
docker compose run --rm -e DATABASE_URL="$DATABASE_URL" backtester --provider synthetic
```

---

## 2. Secrets Manager

One secret holds everything sensitive. The code reads it when `SECRET_ID` is set
(see `trader_core/config.py`), so no key is ever baked into an image.

```bash
cat > secret.json <<EOF
{
  "ALPACA_API_KEY_ID": "PASTE_PAPER_KEY_ID",
  "ALPACA_API_SECRET_KEY": "PASTE_PAPER_SECRET",
  "DATABASE_URL": "$DATABASE_URL",
  "DASHBOARD_PASSWORD": "pick-something"
}
EOF
# edit secret.json to paste your Alpaca keys, then:
SECRET_ARN=$(aws secretsmanager create-secret --name stratos/prod \
  --secret-string file://secret.json --query ARN --output text)
rm secret.json   # it's in AWS now; don't leave credentials lying around (it's git-ignored too)
echo "$SECRET_ARN"
```

---

## 3. ECR repositories

```bash
for r in backtester live-trader dashboard; do
  aws ecr create-repository --repository-name stratos-$r >/dev/null
  # keep only the 5 newest images so storage doesn't pile up
  aws ecr put-lifecycle-policy --repository-name stratos-$r --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep 5","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}' >/dev/null
done
```

## 4. Build and push the images

```bash
./scripts/push_images.sh
```

The script logs Docker in to ECR, builds each image for the right CPU architecture,
and pushes it. Read the comments in it: the `--provenance=false` flag and the
arm64/amd64 split are both things that trip people up.

---

## 5. IAM role for the Lambda functions

A role is the identity a function runs as. This one may write logs and read one secret, nothing else.

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

If `create-function` says the role can't be assumed, IAM hasn't propagated yet. Wait 10 seconds and retry.

The commands below use the four default ETFs in `SYMBOLS`. Paste your own list from `.env` instead
(the whole `SYMBOLS` value, commas and all).

```bash
aws lambda create-function --function-name stratos-backtester \
  --package-type Image --code ImageUri=$REGISTRY/stratos-backtester:latest \
  --role $LAMBDA_ROLE --architectures arm64 --timeout 300 --memory-size 1024 \
  --environment '{"Variables":{"SECRET_ID":"stratos/prod","DATA_PROVIDER":"alpaca","SYMBOLS":"SPY,QQQ,GLD,TLT"}}'

# DRY_RUN=true for the first day: it decides and records but sends no orders.
aws lambda create-function --function-name stratos-live-trader \
  --package-type Image --code ImageUri=$REGISTRY/stratos-live-trader:latest \
  --role $LAMBDA_ROLE --architectures arm64 --timeout 60 --memory-size 512 \
  --environment '{"Variables":{"SECRET_ID":"stratos/prod","SYMBOLS":"SPY,QQQ,GLD,TLT","SIGNAL_MODE":"close","DRY_RUN":"true"}}'

aws lambda wait function-active-v2 --function-name stratos-live-trader
```

Run a backtest in the cloud:

```bash
aws lambda invoke --function-name stratos-backtester \
  --cli-binary-format raw-in-base64-out --cli-read-timeout 310 \
  --payload '{"start": "2021-01-01"}' out.json && cat out.json
```

Poke the live trader (outside market hours it should answer `"market_closed"`):

```bash
aws lambda invoke --function-name stratos-live-trader out.json && cat out.json
aws logs tail /aws/lambda/stratos-live-trader --since 10m
```

When you're happy with a day of dry-run decisions, turn real (paper) orders on.
Note that `--environment` **replaces** all variables, so pass the full set:

```bash
aws lambda update-function-configuration --function-name stratos-live-trader \
  --environment '{"Variables":{"SECRET_ID":"stratos/prod","SYMBOLS":"SPY,QQQ,GLD,TLT","SIGNAL_MODE":"close","DRY_RUN":"false"}}'
```

---

## 7. EventBridge Scheduler: every 5 minutes, weekdays

The schedule is written in New York time, so it follows daylight saving automatically.
It fires from 9:00 to 15:55; the function checks Alpaca's market clock and exits
early before the 9:30 open and on market holidays.

```bash
aws iam create-role --role-name stratos-scheduler --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

LIVE_ARN=$(aws lambda get-function --function-name stratos-live-trader --query Configuration.FunctionArn --output text)
aws iam put-role-policy --role-name stratos-scheduler --policy-name invoke-live-trader \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"$LIVE_ARN\"}]}"
SCHED_ROLE=$(aws iam get-role --role-name stratos-scheduler --query Role.Arn --output text)

# The Input passes the scheduled time into the function; a retried invocation gets the
# same value, so its orders reuse the same client_order_id and Alpaca rejects the repeat.
cat > target.json <<EOF
{
  "Arn": "$LIVE_ARN",
  "RoleArn": "$SCHED_ROLE",
  "Input": "{\"scheduled_time\": \"<aws.scheduler.scheduled-time>\"}"
}
EOF

aws scheduler create-schedule --name stratos-every-5-min \
  --schedule-expression "cron(0/5 9-15 ? * MON-FRI *)" \
  --schedule-expression-timezone "America/New_York" \
  --flexible-time-window '{"Mode":"OFF"}' \
  --target file://target.json
rm target.json
```

To pause trading, disable the schedule in the EventBridge Scheduler console (one toggle).

---

## 8. Dashboard: ECS Express Mode

**Why not App Runner?** The original plan used App Runner, but AWS closed it to new
customers on April 30, 2026. AWS now points people to **Amazon ECS Express Mode**,
which from one container image creates an ECS service on Fargate, an Application
Load Balancer, auto scaling and networking. It's also the more useful thing to have
on a résumé, since ECS/Fargate is what a lot of companies actually run.

**Cost warning.** The load balancer and the Fargate task both bill by the hour even
when nobody is looking. On the new-account free plan this comes out of your credits,
but it's the part of the project most worth deleting when you're not using it.
The zero-cost alternative is to run the dashboard on your laptop against RDS:

```bash
docker compose run --rm -p 8501:8080 -e DATABASE_URL="$DATABASE_URL" dashboard
# then open http://localhost:8501
```

To deploy it, the console is easiest the first time because it can create the two IAM
roles for you:

1. ECS console → **Express mode** → Create.
2. Image URI: `<ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/stratos-dashboard:latest`
3. Container port: `8080`. Health check path: `/_stcore/health`
   (the default `/ping` doesn't exist in Streamlit, so the load balancer would keep killing the task).
4. Task execution role and infrastructure role: **Create new role** for each.
5. Secrets: add `DATABASE_URL` with value-from `<SECRET_ARN>:DATABASE_URL::` and
   `DASHBOARD_PASSWORD` with `<SECRET_ARN>:DASHBOARD_PASSWORD::`
   (the `:KEY::` suffix pulls one field out of the JSON secret).
6. Give the new **task execution role** permission to read the secret, or the task
   will fail to start with an access-denied error in the service events:

   ```bash
   aws iam put-role-policy --role-name <EXECUTION_ROLE_NAME> --policy-name read-trading-secret \
     --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"$SECRET_ARN\"}]}"
   ```

7. Create, wait for the service to become healthy, and open the URL it gives you.

After the first time you can script it with `aws ecs create-express-gateway-service`
(see `aws ecs create-express-gateway-service help` for the `--primary-container`,
`--health-check-path`, `--execution-role-arn` and `--infrastructure-role-arn` options).

---

## 9. Costs and tearing down

Running 24/7: the RDS instance, the dashboard's load balancer and Fargate task, and
the secret (about $0.40/month). Everything else is pay-per-use and tiny at this volume:
about 84 Lambda runs per trading day, plus a few MB of images in ECR. New AWS accounts
get a credit-based free plan (currently $100 at sign-up, up to $100 more, for 6 months);
check **Billing → Free tier** and your budget alert rather than trusting any number here.

To stop the database for a while without deleting it: `aws rds stop-db-instance --db-instance-identifier stratos-db`
(AWS restarts stopped instances automatically after 7 days).

Delete everything:

```bash
aws scheduler delete-schedule --name stratos-every-5-min
aws lambda delete-function --function-name stratos-live-trader
aws lambda delete-function --function-name stratos-backtester
# ECS Express service: delete it in the ECS console (this also removes its load balancer)
aws rds delete-db-instance --db-instance-identifier stratos-db --skip-final-snapshot
aws rds wait db-instance-deleted --db-instance-identifier stratos-db
aws ec2 delete-security-group --group-id $DB_SG
aws secretsmanager delete-secret --secret-id stratos/prod --force-delete-without-recovery
for r in backtester live-trader dashboard; do aws ecr delete-repository --repository-name stratos-$r --force; done
aws iam delete-role-policy --role-name stratos-scheduler --policy-name invoke-live-trader
aws iam delete-role --role-name stratos-scheduler
aws iam delete-role-policy --role-name stratos-lambda --policy-name read-trading-secret
aws iam detach-role-policy --role-name stratos-lambda --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name stratos-lambda
```
