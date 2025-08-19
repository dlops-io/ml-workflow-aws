import os
import random
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow.model import TensorFlowModel

# Get AWS environment variables
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


def _resolve_lambda_role():
    # Derive Lambda execution role ARN using account ID
    try:
        sts_client = boto3.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        role_arn = f"arn:aws:iam::{account_id}:role/CheeseAppLambdaExecutionRole"
        print(f"Using derived Lambda execution role: {role_arn}")
        return role_arn
    except Exception as e:
        raise RuntimeError(
            "Unable to derive Lambda execution role. Please ensure CheeseAppLambdaExecutionRole exists in your AWS account."
        ) from e


def model_training(
    location: str = "",
    staging_bucket: str = "",
    bucket_name: str = "",
    epochs: int = 15,
    batch_size: int = 16,
    model_name: str = "mobilenetv2",
    train_base: bool = False,
):
    print("Model Training Job")

    # Region-aware sessions
    boto_sess = boto3.Session(region_name=AWS_REGION)
    sm_sess = sagemaker.Session(boto_session=boto_sess)
    role = _resolve_role()

    job_name = f"cheese-{generate_uuid()}"

    # Build hyperparameters dict - only include train_base if True
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "model_name": model_name,
        "bucket_name": bucket_name,
    }
    
    # Only add train_base if it's True (argparse store_true behavior)
    if train_base:
        hyperparameters["train_base"] = ""  # Empty value for store_true action

    estimator = TensorFlow(
        entry_point="package/trainer/task.py",
        source_dir=f"{S3_PACKAGE_URI}/cheese-app-trainer.tar.gz",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        framework_version="2.13",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=f"s3://{bucket_name}/model-output",
        sagemaker_session=sm_sess,
    )

    print(f"Starting training job: {job_name}")
    print(f"Using source code from: {S3_PACKAGE_URI}")
    
    training_input = f"s3://{bucket_name}/tfrecords.zip"
    estimator.fit(inputs=training_input, job_name=job_name, wait=True)

    return estimator


def model_deploy(
    bucket_name: str = "",
):
    print("Model Deploy Job")

    # Region-aware sessions
    boto_sess = boto3.Session(region_name=AWS_REGION)
    sm_sess = sagemaker.Session(boto_session=boto_sess)
    role = _resolve_role()

    # Find the latest training job output directory by creation time
    s3 = boto3.client('s3')
    
    try:
        # List all job directories under model-output/
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix='model-output/',
            Delimiter='/'
        )
        
        if 'CommonPrefixes' in response:
            # Get all job directories that start with 'cheese-' and their creation times
            job_dirs = []
            for prefix in response['CommonPrefixes']:
                dir_name = prefix['Prefix'].split('/')[-2]  # Get directory name without trailing slash
                if dir_name.startswith('cheese-'):
                    # Get the creation time by looking at a file inside the directory
                    try:
                        objects_response = s3.list_objects_v2(
                            Bucket=bucket_name,
                            Prefix=f'model-output/{dir_name}/',
                            MaxKeys=1
                        )
                        if 'Contents' in objects_response and objects_response['Contents']:
                            creation_time = objects_response['Contents'][0]['LastModified']
                            job_dirs.append((dir_name, creation_time))
                    except Exception as e:
                        print(f"Warning: Could not get creation time for {dir_name}: {e}")
            
            if not job_dirs:
                raise ValueError("No training job outputs found with 'cheese-' prefix")
            
            # Sort by creation time (most recent first)
            job_dirs.sort(key=lambda x: x[1], reverse=True)
            latest_job = job_dirs[0][0]  # Get the directory name of the most recent job
            
            model_data = f"s3://{bucket_name}/model-output/{latest_job}/output/model.tar.gz"
            print(f"Found latest training job: {latest_job}")
            print(f"Using model data: {model_data}")
            
        else:
            raise ValueError("No training job outputs found in model-output directory")
            
    except Exception as e:
        print(f"Error finding model data: {e}")
        raise

    # Create TensorFlow model
    model = TensorFlowModel(
        model_data=model_data,
        role=role,
        framework_version="2.13",
        sagemaker_session=sm_sess,
    )

    # Deploy model to endpoint
    endpoint_name = f"cheese-app-endpoint-{generate_uuid()}"
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=endpoint_name,
    )
    
    print(f"Model deployed to endpoint: {endpoint_name}")
    return predictor
