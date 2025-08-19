#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="cheese-app-data-processor"
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export S3_BUCKET_NAME="cheese-app-ml-workflow-demo"

# Auto-detect AWS credentials from ~/.aws/credentials
if command -v aws &> /dev/null; then
    export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
    export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
    export AWS_DEFAULT_REGION=$(aws configure get region)
    
    if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
        echo "AWS not configured. Run: aws configure"
        exit 1
    fi

else
    echo "AWS CLI not installed. Install with: brew install awscli"
    exit 1
fi

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .
# docker build -t $IMAGE_NAME --platform=linux/amd64/v2 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$PERSISTENT_DIR":/persistent \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
-e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
-e S3_BUCKET_NAME=${S3_BUCKET_NAME} \
$IMAGE_NAME
