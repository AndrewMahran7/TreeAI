# ArborNote AWS Deployment Checklist

## Pre-Deployment Checklist

### AWS Account Setup
- [ ] AWS account created and verified
- [ ] AWS CLI installed and configured
- [ ] AWS credentials configured with appropriate permissions
- [ ] Billing alerts set up
- [ ] EC2 key pair created and downloaded
- [ ] Domain name registered (optional but recommended)

### Local Environment
- [ ] Terraform installed (version >= 1.0)
- [ ] Docker installed and running
- [ ] Git repository set up and pushed to GitHub
- [ ] All sensitive files (.env, *.key, *.pem) added to .gitignore
- [ ] Project files committed and pushed

### Configuration Files
- [ ] `.env` file created from `.env.production` template
- [ ] `FLASK_SECRET_KEY` generated (minimum 32 characters)
- [ ] `NEARMAP_API_KEY` obtained and configured
- [ ] `terraform.tfvars` created and configured
- [ ] Domain name configured in variables (if applicable)
- [ ] SSL certificate requested in AWS Certificate Manager (if using domain)

## Deployment Checklist

### Infrastructure Deployment
- [ ] Navigate to `aws-infrastructure/` directory
- [ ] Run `terraform init`
- [ ] Review `terraform plan` output
- [ ] Run `terraform apply` and confirm
- [ ] Note load balancer DNS name from output
- [ ] Verify all resources created successfully in AWS Console

### Application Verification
- [ ] Wait 10-15 minutes for full deployment
- [ ] Test health endpoint: `curl http://[load-balancer-dns]/health`
- [ ] Access web interface: `http://[load-balancer-dns]`
- [ ] Test basic functionality (upload KML, start processing)
- [ ] Verify logs are being written to CloudWatch
- [ ] Check Auto Scaling Group shows healthy instances

### Security Configuration
- [ ] Restrict SSH access in security groups to your IP only
- [ ] Verify S3 bucket has public access blocked
- [ ] Enable AWS CloudTrail for audit logging
- [ ] Set up AWS Config for compliance monitoring
- [ ] Configure AWS GuardDuty for threat detection (optional)
- [ ] Review IAM permissions and apply principle of least privilege

## Post-Deployment Checklist

### Monitoring Setup
- [ ] CloudWatch dashboards configured
- [ ] Cost monitoring alerts set up
- [ ] Performance monitoring alerts configured
- [ ] Log aggregation working properly
- [ ] Health check alerts configured
- [ ] Error tracking configured (Sentry if enabled)

### DNS and SSL (if using custom domain)
- [ ] DNS records pointing to load balancer
- [ ] SSL certificate validated and issued
- [ ] HTTPS redirect working
- [ ] SSL Labs rating A or higher
- [ ] Certificate auto-renewal configured

### Testing and Validation
- [ ] Upload and process sample KML files
- [ ] Test file downloads (CSV, JSON, KML outputs)
- [ ] Verify geofence processing works correctly
- [ ] Test with different imagery dates
- [ ] Load test with multiple concurrent requests
- [ ] Test failure scenarios (invalid files, large areas)

### Documentation and Access
- [ ] Document load balancer DNS or domain name
- [ ] Save EC2 key pair in secure location
- [ ] Document SSH access procedures
- [ ] Create runbook for common operations
- [ ] Share access credentials with team (securely)
- [ ] Document backup and recovery procedures

## Ongoing Maintenance Checklist

### Daily
- [ ] Check AWS costs in billing dashboard
- [ ] Review CloudWatch logs for errors
- [ ] Verify application health endpoint

### Weekly
- [ ] Review application performance metrics
- [ ] Check for security alerts in AWS Console
- [ ] Review resource utilization
- [ ] Test backup procedures

### Monthly
- [ ] Review and optimize costs
- [ ] Update application dependencies
- [ ] Review security groups and access permissions
- [ ] Test disaster recovery procedures
- [ ] Review and update documentation

### Quarterly
- [ ] Security audit and penetration testing
- [ ] Review compliance requirements
- [ ] Update disaster recovery plan
- [ ] Team training on operations procedures

## Troubleshooting Checklist

### Application Not Responding
- [ ] Check Auto Scaling Group health
- [ ] Review EC2 instance logs
- [ ] Check Docker container status
- [ ] Verify security group rules
- [ ] Check load balancer target health

### High Costs
- [ ] Review CloudWatch metrics for unusual traffic
- [ ] Check EC2 instance types and utilization
- [ ] Review S3 storage usage
- [ ] Analyze data transfer costs
- [ ] Consider reserved instances for predictable workloads

### Performance Issues
- [ ] Monitor CPU and memory usage
- [ ] Check database performance (if applicable)
- [ ] Review application logs for bottlenecks
- [ ] Consider scaling up or out
- [ ] Optimize Docker images and container resources

### Security Concerns
- [ ] Review CloudTrail logs for unusual activity
- [ ] Check security group configurations
- [ ] Verify SSL certificate status
- [ ] Review access patterns in logs
- [ ] Update security patches

## Emergency Response

### Service Outage
1. Check AWS Service Health Dashboard
2. Review Auto Scaling Group and EC2 health
3. Check load balancer target health
4. Review recent deployments or changes
5. Scale up instances if needed
6. Contact AWS Support if infrastructure issue

### Security Incident
1. Isolate affected resources
2. Preserve logs and evidence
3. Notify stakeholders
4. Follow incident response procedures
5. Document lessons learned
6. Update security measures

### Data Loss or Corruption
1. Stop application to prevent further damage
2. Restore from latest backup
3. Verify data integrity
4. Test application functionality
5. Document root cause
6. Update backup procedures

## Success Criteria

### Technical
- [ ] Application accessible from internet
- [ ] Health checks passing consistently
- [ ] Auto scaling working properly
- [ ] Logs and monitoring operational
- [ ] SSL/TLS configured correctly
- [ ] Backup and recovery tested

### Business
- [ ] Cost within expected budget
- [ ] Performance meets requirements
- [ ] Security requirements satisfied
- [ ] Team can operate and maintain system
- [ ] Documentation complete and accessible
- [ ] Disaster recovery plan validated

---

**Note**: Keep this checklist updated as your deployment evolves and requirements change. Regular reviews ensure continued operational excellence.