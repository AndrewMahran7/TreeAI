# terraform.tfvars.example
# Copy this file to terraform.tfvars and update with your values

# AWS Configuration
aws_region = "us-east-1"

# Project Configuration
project_name = "interface"
environment  = "prod"

# Domain Configuration (optional - leave empty if not using custom domain)
domain_name = ""  # e.g., "yourdomain.com"

# EC2 Configuration
instance_type = "t3.medium"  # Adjust based on your needs
key_name      = "interface-key.pem"  # Must exist in your AWS account

# Application Configuration
nearmap_api_key  = "ZTBkNjI1NDYtZmVhMy00MDA0LTk4NDUtZGNkYzY2MzBmNzg2"
flask_secret_key = "myflaskrizzysuperduperrizzkeythatisalsoastring"