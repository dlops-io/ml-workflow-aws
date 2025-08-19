#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

export IMAGE_NAME="cheese-app-workflow"
export BASE_DIR=$(pwd)
export S3_BUCKET_NAME="cheese-app-ml-workflow-demo"
export S3_PACKAGE_URI="s3://cheese-app-ml-workflow-demo"

# Auto-detect AWS credentials from ~/.aws/credentials
if command -v aws &> /dev/null; then
    export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
    export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
    export AWS_REGION=$(aws configure get region)
    
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
docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .


# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
-v "$BASE_DIR/../data-collector":/data-collector \
-v "$BASE_DIR/../data-processor":/data-processor \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
-e AWS_REGION=${AWS_REGION} \
-e S3_BUCKET_NAME=${S3_BUCKET_NAME} \
-e S3_PACKAGE_URI=${S3_PACKAGE_URI} \
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME
