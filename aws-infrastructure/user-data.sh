#!/bin/bash
# ArborNote EC2 User Data Script
# This script sets up the environment and starts the application

set -e

# Log everything
exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting ArborNote setup at $(date)"

# Update the system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
yum install -y git

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf awscliv2.zip aws/

# Create application directory
mkdir -p /opt/arbornote
cd /opt/arbornote

# Clone the repository (you'll need to replace this with your actual repo)
# For now, we'll create the structure and copy files
git clone https://github.com/AndrewMahran7/TreeAI.git .

# Set up environment variables
cat > .env << EOF
FLASK_ENV=production
FLASK_SECRET_KEY=${flask_secret_key}
NEARMAP_API_KEY=${nearmap_api_key}
AWS_DEFAULT_REGION=${aws_region}
AWS_S3_BUCKET=${s3_bucket}
MAX_CONCURRENT_JOBS=2
JOB_TIMEOUT_MINUTES=30
LOG_LEVEL=INFO
EOF

# Set up directory permissions
chown -R ec2-user:ec2-user /opt/arbornote
chmod +x /opt/arbornote/*.sh 2>/dev/null || true

# Create systemd service for the application
cat > /etc/systemd/system/arbornote.service << EOF
[Unit]
Description=ArborNote Tree Detection Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/arbornote
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=ec2-user
Group=ec2-user

[Install]
WantedBy=multi-user.target
EOF

# Create log rotation for application logs
cat > /etc/logrotate.d/arbornote << EOF
/opt/arbornote/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 644 ec2-user ec2-user
    postrotate
        /usr/local/bin/docker-compose -f /opt/arbornote/docker-compose.yml exec arbornote kill -USR1 1 2>/dev/null || true
    endscript
}
EOF

# Install CloudWatch agent
yum install -y amazon-cloudwatch-agent

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "cwagent"
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/opt/arbornote/logs/arbornote.log",
                        "log_group_name": "/aws/ec2/arbornote/application",
                        "log_stream_name": "{instance_id}/application.log",
                        "retention_in_days": 14
                    },
                    {
                        "file_path": "/var/log/user-data.log",
                        "log_group_name": "/aws/ec2/arbornote/system",
                        "log_stream_name": "{instance_id}/user-data.log",
                        "retention_in_days": 7
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "ArborNote/EC2",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

# Build and start the application
cd /opt/arbornote
sudo -u ec2-user docker-compose build
sudo -u ec2-user docker-compose up -d

# Enable and start the service
systemctl daemon-reload
systemctl enable arbornote.service
systemctl start arbornote.service

# Set up automatic updates
cat > /etc/cron.d/arbornote-update << EOF
# Update ArborNote daily at 2 AM
0 2 * * * ec2-user cd /opt/arbornote && git pull && docker-compose build --no-cache && docker-compose up -d
EOF

# Create health check script
cat > /opt/arbornote/health-check.sh << 'EOF'
#!/bin/bash
# Health check script for ArborNote

HEALTH_URL="http://localhost:4000/health"
MAX_RETRIES=3
RETRY_DELAY=10

for i in $(seq 1 $MAX_RETRIES); do
    if curl -f $HEALTH_URL > /dev/null 2>&1; then
        echo "Health check passed"
        exit 0
    fi
    
    echo "Health check failed (attempt $i/$MAX_RETRIES)"
    if [ $i -lt $MAX_RETRIES ]; then
        sleep $RETRY_DELAY
    fi
done

echo "Health check failed after $MAX_RETRIES attempts"
# Restart the service
systemctl restart arbornote.service
exit 1
EOF

chmod +x /opt/arbornote/health-check.sh

# Set up health check cron job
cat > /etc/cron.d/arbornote-health << EOF
# Health check every 5 minutes
*/5 * * * * ec2-user /opt/arbornote/health-check.sh
EOF

# Wait for application to start
sleep 60

# Final health check
curl -f http://localhost:4000/health || echo "Warning: Application may not be ready yet"

echo "ArborNote setup completed at $(date)"