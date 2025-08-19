#!/bin/bash

echo "Container is running!!!"

# AWS configuration (credentials should be set via environment variables or IAM roles)
echo "AWS Region: $AWS_REGION"
echo "S3 Bucket: $S3_BUCKET_NAME"
echo "Package URI: $S3_PACKAGE_URI"

# Test AWS connectivity
aws sts get-caller-identity

#/bin/bash
pipenv shell
