"""
CLI to kick off SageMaker training.
Usage:
    python cli.py --train
"""

import os
import argparse
import random
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow

AWS_REGION = os.environ["AWS_REGION"]          
S3_PACKAGE_URI = os.environ["S3_PACKAGE_URI"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]


def generate_uuid(length: int = 8) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choices(alphabet, k=length))


def _resolve_role():
    # First try to get execution role from SageMaker context
    try:
        return sagemaker.get_execution_role()
    except Exception:
        # If not in SageMaker context, derive role ARN using account ID
        try:
            sts_client = boto3.client('sts')
            account_id = sts_client.get_caller_identity()['Account']
            role_arn = f"arn:aws:iam::{account_id}:role/CheeseAppSageMakerExecutionRole"
            print(f"Using derived SageMaker execution role: {role_arn}")
            return role_arn
        except Exception as e:
            raise RuntimeError(
                "Unable to derive SageMaker execution role. Please ensure CheeseAppSageMakerExecutionRole exists in your AWS account."
            ) from e


def main(args=None):
    if args.train:
        print("Train Model")

        # Region-aware sessions
        boto_sess = boto3.Session(region_name=AWS_REGION)
        sm_sess = sagemaker.Session(boto_session=boto_sess)
        role = _resolve_role()

        job_name = f"cheese-{generate_uuid()}"

        estimator = TensorFlow(
            entry_point="package/trainer/task.py",
            source_dir=f"{S3_PACKAGE_URI}/cheese-app-trainer.tar.gz",
            role=role,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            framework_version="2.13",
            py_version="py310",
            hyperparameters={
                "epochs": 15,
                "batch_size": 16,
                "bucket_name": S3_BUCKET_NAME,
            },
            output_path=f"s3://{S3_BUCKET_NAME}/model-output",
            sagemaker_session=sm_sess,
        )

        print(f"Starting training job: {job_name}")
        print(f"Using source code from: {S3_PACKAGE_URI}")
        
        training_input = f"s3://{S3_BUCKET_NAME}/tfrecords.zip"
        estimator.fit(inputs=training_input, job_name=job_name, wait=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training CLI")
    parser.add_argument("-t", "--train", action="store_true", help="Train model")
    args = parser.parse_args()
    main(args)
