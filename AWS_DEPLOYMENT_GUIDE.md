# ArborNote AWS Deployment Guide

This guide will help you deploy your ArborNote Tree Detection application to AWS for private hosting, accessible from any computer via IP address while your local computer is turned off.

## 🏗️ Architecture Overview

Your application will be deployed using:
- **EC2 instances** running Docker containers
- **Application Load Balancer** for high availability and SSL termination
- **Auto Scaling Group** for automatic scaling
- **S3 bucket** for file storage
- **VPC** with public subnets for security
- **CloudWatch** for monitoring and logs

## 📋 Prerequisites

### 1. AWS Account Setup
1. Create an AWS account at [aws.amazon.com](https://aws.amazon.com)
2. Install AWS CLI: [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
3. Configure AWS credentials:
   ```bash
   aws configure
   ```
   Enter your Access Key ID, Secret Access Key, default region (`us-east-1`), and output format (`json`)

### 2. Required Software
- **Terraform**: [Install Terraform](https://developer.hashicorp.com/terraform/downloads)
- **Git**: [Install Git](https://git-scm.com/downloads)
- **Docker** (for local testing): [Install Docker](https://docs.docker.com/get-docker/)

### 3. AWS Resources to Create
- EC2 Key Pair for SSH access
- Domain name (optional, but recommended for production)

## 🔧 Setup Steps

### Step 1: Prepare Your Project

1. **Clone or prepare your project files**:
   ```bash
   cd /path/to/your/project
   git add .
   git commit -m "Prepare for AWS deployment"
   git push origin main
   ```

2. **Update your GitHub repository** (if private, make it public temporarily or set up deploy keys)

### Step 2: Create EC2 Key Pair

1. Go to AWS Console → EC2 → Key Pairs
2. Click "Create key pair"
3. Name: `arbornote-key`
4. Type: RSA
5. Format: `.pem`
6. Download and save securely

### Step 3: Configure Environment Variables

1. **Copy the environment template**:
   ```bash
   cp .env.production .env
   ```

2. **Update `.env` with your values**:
   ```bash
   # Required - Get from Nearmap
   NEARMAP_API_KEY=your-nearmap-api-key-here
   
   # Required - Generate a strong secret
   FLASK_SECRET_KEY=your-super-secret-key-minimum-32-characters
   
   # Optional - For error tracking
   SENTRY_DSN=your-sentry-dsn
   
   # Domain (if you have one)
   ALLOWED_HOSTS=your-domain.com,www.your-domain.com
   ```

### Step 4: Deploy Infrastructure with Terraform

1. **Navigate to infrastructure directory**:
   ```bash
   cd aws-infrastructure
   ```

2. **Copy and configure variables**:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

3. **Edit `terraform.tfvars`**:
   ```hcl
   # AWS Configuration
   aws_region = "us-east-1"
   
   # Project Configuration
   project_name = "arbornote"
   environment  = "prod"
   
   # EC2 Configuration
   instance_type = "t3.medium"
   key_name      = "arbornote-key"  # The key pair you created
   
   # Application Configuration
   nearmap_api_key  = "your-nearmap-api-key"
   flask_secret_key = "your-super-secret-flask-key"
   ```

4. **Initialize and deploy**:
   ```bash
   # Initialize Terraform
   terraform init
   
   # Review the deployment plan
   terraform plan
   
   # Deploy (this takes 5-10 minutes)
   terraform apply
   ```

5. **Note the outputs**:
   ```
   load_balancer_dns = "arbornote-alb-xxxxxxxxx.us-east-1.elb.amazonaws.com"
   s3_bucket_name = "arbornote-app-data-xxxxxxxx"
   ```

### Step 5: Verify Deployment

1. **Wait for deployment** (5-10 minutes for full startup)

2. **Check application health**:
   ```bash
   curl http://your-load-balancer-dns/health
   ```

3. **Access your application**:
   - Open browser to: `http://your-load-balancer-dns`
   - You should see the ArborNote interface

### Step 6: Set Up Domain (Optional but Recommended)

If you have a domain name:

1. **Update Route 53** (or your DNS provider):
   - Create an A record pointing to your load balancer
   - Example: `arbornote.yourdomain.com` → `your-load-balancer-dns`

2. **Set up SSL Certificate**:
   ```bash
   # Request certificate in AWS Certificate Manager
   aws acm request-certificate \
     --domain-name arbornote.yourdomain.com \
     --validation-method DNS \
     --region us-east-1
   ```

3. **Update ALB to use HTTPS** (advanced - requires additional Terraform configuration)

## 💰 Cost Estimation

### Monthly AWS Costs (approximate):
- **t3.medium EC2**: ~$30/month
- **Application Load Balancer**: ~$18/month
- **S3 Storage**: ~$1-5/month (depending on usage)
- **Data Transfer**: ~$5-15/month (depending on traffic)
- **CloudWatch Logs**: ~$1-3/month

**Total: ~$55-75/month**

### Cost Optimization Tips:
- Use `t3.small` for lighter workloads ($15/month)
- Consider Reserved Instances for 30-70% savings
- Set up CloudWatch alarms for cost monitoring

## 🔒 Security Features Included

- **VPC isolation** with public/private subnets
- **Security Groups** restricting access to necessary ports only
- **IAM roles** with minimal required permissions
- **S3 bucket** with encryption and public access blocked
- **Auto Scaling** for availability
- **CloudWatch monitoring** for security events

## 📊 Monitoring and Maintenance

### Built-in Monitoring:
- **Health checks** every 30 seconds
- **Auto-restart** if application fails
- **CloudWatch logs** for debugging
- **CloudWatch metrics** for performance

### Regular Maintenance:
1. **Monitor costs** in AWS Console
2. **Check logs** in CloudWatch
3. **Update application** by pushing to Git (auto-deploys daily at 2 AM)
4. **Backup model files** regularly

## 🔧 Troubleshooting

### Common Issues:

1. **Application not responding**:
   ```bash
   # SSH into EC2 instance
   ssh -i arbornote-key.pem ec2-user@your-instance-ip
   
   # Check application status
   sudo systemctl status arbornote
   docker-compose logs
   ```

2. **High costs**:
   - Check CloudWatch for unusual traffic
   - Consider smaller instance types
   - Review S3 storage usage

3. **SSL certificate issues**:
   - Verify domain ownership
   - Check DNS propagation
   - Review ACM certificate status

### Getting Help:
- **AWS Support**: Available with paid support plans
- **Application Logs**: Check `/opt/arbornote/logs/` on EC2
- **CloudWatch Logs**: Monitor real-time application behavior

## 🚀 Scaling and Upgrades

### Horizontal Scaling:
- Auto Scaling Group will automatically add/remove instances
- Modify `min_size` and `max_size` in Terraform for more capacity

### Vertical Scaling:
- Update `instance_type` in `terraform.tfvars`
- Run `terraform apply` to update

### Application Updates:
- Push changes to Git repository
- Application updates automatically daily at 2 AM
- For immediate updates: SSH to instance and run `docker-compose pull && docker-compose up -d`

## 📝 Important Notes

1. **Save your EC2 key pair** - you cannot recover it if lost
2. **Monitor your AWS costs** regularly
3. **Keep your Nearmap API key secure** - never commit to public repositories
4. **Regular backups** of your model checkpoints to S3
5. **Consider enabling AWS CloudTrail** for audit logging

## 🎯 Next Steps After Deployment

1. **Test thoroughly** with your actual KML files and workflows
2. **Set up monitoring alerts** for high costs or failures
3. **Create backup procedures** for your model files
4. **Consider CDN** (CloudFront) for better global performance
5. **Implement user authentication** if needed for security

## 🔄 Updating the Application

When you want to update your application:

1. **Make changes locally**
2. **Push to your Git repository**
3. **Changes automatically deploy daily**, or force update:
   ```bash
   ssh -i arbornote-key.pem ec2-user@your-instance-ip
   cd /opt/arbornote
   git pull
   docker-compose build --no-cache
   docker-compose up -d
   ```

## 🛡️ Security Best Practices

1. **Restrict SSH access** to your IP only in security groups
2. **Regularly update dependencies** in requirements.prod.txt
3. **Monitor access logs** in CloudWatch
4. **Use AWS Secrets Manager** for sensitive configuration (advanced)
5. **Enable AWS GuardDuty** for threat detection (additional cost)

---

Your ArborNote application is now running on AWS! You can access it from anywhere using the load balancer DNS name, and it will remain available even when your local computer is off.

For support or questions about this deployment, check the AWS documentation or consider AWS Support for production workloads.