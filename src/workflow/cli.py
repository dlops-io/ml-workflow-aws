"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from model import model_training as model_training_job, model_deploy as model_deploy_job, _resolve_role, _resolve_lambda_role
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow.model import TensorFlowModel


# AWS Environment variables
AWS_REGION = os.environ["AWS_REGION"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_PACKAGE_URI = os.environ["S3_PACKAGE_URI"]

# Get AWS Account ID for ECR
sts_client = boto3.client('sts')
ACCOUNT_ID = sts_client.get_caller_identity()['Account']

# ECR Docker images
DATA_COLLECTOR_IMAGE = f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/cheese-app-data-collector:latest"
DATA_PROCESSOR_IMAGE = f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/cheese-app-data-processor:latest"


def generate_uuid(length: int = 8) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choices(alphabet, k=length))


def data_collector():
    print("data_collector()")

    # Region-aware sessions
    boto_sess = boto3.Session(region_name=AWS_REGION)
    sm_sess = sagemaker.Session(boto_session=boto_sess)
    role = _resolve_role()

    processor = Processor(
        image_uri=DATA_COLLECTOR_IMAGE,
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        entrypoint=["python3", "/app/cli.py"],
        sagemaker_session=sm_sess,
    )

    job_name = f"cheese-app-data-collector-{generate_uuid()}"

    processor.run(
        arguments=[
            "--search",
            "--nums","10",
            "--query","brie cheese","gouda cheese","gruyere cheese","parmigiano cheese",
            "--bucket", S3_BUCKET_NAME,
        ],
        outputs=[
            ProcessingOutput(
                output_name="raw-data",
                source="/opt/ml/processing/output",
                destination=f"s3://{S3_BUCKET_NAME}/raw",
            )
        ],
        job_name=job_name,
        logs=True,
        wait=True,
    )
    


def data_processor():
    print("data_processor()")

    # Region-aware sessions
    boto_sess = boto3.Session(region_name=AWS_REGION)
    sm_sess = sagemaker.Session(boto_session=boto_sess)
    role = _resolve_role()

    processor = Processor(
        image_uri=DATA_PROCESSOR_IMAGE,
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        entrypoint=["python3", "/app/cli.py"],
        sagemaker_session=sm_sess,
    )

    job_name = f"cheese-app-data-processor-{generate_uuid()}"

    processor.run(
        arguments=[
            "--clean",
            "--prepare",
            "--bucket", S3_BUCKET_NAME,
        ],
        inputs=[
            ProcessingInput(
                input_name="raw-data",
                source=f"s3://{S3_BUCKET_NAME}/raw",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="clean-data",
                source="/opt/ml/processing/output/clean",
                destination=f"s3://{S3_BUCKET_NAME}/clean",
            ),
            ProcessingOutput(
                output_name="tfrecords",
                source="/opt/ml/processing/output/tfrecords",
                destination=f"s3://{S3_BUCKET_NAME}/tfrecords",
            )
        ],
        job_name=job_name,
        logs=True,
        wait=True,
    )
    


def model_training():
    print("model_training()")

    # Call the model training function directly
    estimator = model_training_job(
        location=AWS_REGION,
        staging_bucket=S3_PACKAGE_URI,
        bucket_name=S3_BUCKET_NAME,
    )



def model_deploy():
    print("model_deploy()")

    # Call the model deploy function directly
    predictor = model_deploy_job(
        bucket_name=S3_BUCKET_NAME,
    )



def pipeline():
    """Run the same flow as a SageMaker Pipeline to get a DAG + events in Unified Studio."""
    print("Defining SageMaker Pipeline")

    boto_sess = boto3.Session(region_name=AWS_REGION)
    sm_pipe_sess = PipelineSession(boto_session=boto_sess)
    role = _resolve_role()

    cache_cfg = CacheConfig(enable_caching=True, expire_after="30d")

    # -----------------------
    # Step 1: Data Collection
    # -----------------------
    data_collector = Processor(
        image_uri=DATA_COLLECTOR_IMAGE,
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        entrypoint=["python3", "/app/cli.py"],
        sagemaker_session=sm_pipe_sess,
        env={"AWS_REGION": AWS_REGION, "S3_BUCKET_NAME": S3_BUCKET_NAME},
    )

    step_collect = ProcessingStep(
        name="DataCollector",
        processor=data_collector,
        job_arguments=[
            "--search",
            "--nums", "50",
            "--query", "brie cheese", "gouda cheese", "gruyere cheese", "parmigiano cheese",
            "--bucket", S3_BUCKET_NAME,
        ],
        outputs=[
            ProcessingOutput(
                output_name="raw-data",
                source="/opt/ml/processing/output",
                destination=f"s3://{S3_BUCKET_NAME}/raw",
            )
        ],
        cache_config=cache_cfg,
    )

    # ----------------------
    # Step 2: Data Processor
    # ----------------------
    data_processor = Processor(
        image_uri=DATA_PROCESSOR_IMAGE,
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        entrypoint=["python3", "/app/cli.py"],
        sagemaker_session=sm_pipe_sess,
        env={"AWS_REGION": AWS_REGION, "S3_BUCKET_NAME": S3_BUCKET_NAME},
    )

    step_process = ProcessingStep(
        name="DataProcessor",
        processor=data_processor,
        job_arguments=["--clean", "--prepare", "--bucket", S3_BUCKET_NAME],
        inputs=[
            ProcessingInput(
                input_name="raw-data",
                source=step_collect.properties.ProcessingOutputConfig
                    .Outputs["raw-data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="clean-data",
                source="/opt/ml/processing/output/clean",
                destination=f"s3://{S3_BUCKET_NAME}/clean",
            ),
            ProcessingOutput(
                output_name="tfrecords",
                source="/opt/ml/processing/output/tfrecords",
                destination=f"s3://{S3_BUCKET_NAME}/tfrecords",
            ),
        ],
        cache_config=cache_cfg,
    )

    # ----------------
    # Step 3: Training
    # ----------------
    hyperparameters = {
        "epochs": 15,
        "batch_size": 16,
        "model_name": "mobilenetv2",
        "bucket_name": S3_BUCKET_NAME,
    }
    
    estimator = TensorFlow(
        entry_point="package/trainer/task.py",
        source_dir=f"{S3_PACKAGE_URI}/cheese-app-trainer.tar.gz",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        framework_version="2.13",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=f"s3://{S3_BUCKET_NAME}/model-output",
        sagemaker_session=sm_pipe_sess,
    )

    step_train = TrainingStep(
        name="ModelTraining",
        estimator=estimator,
        inputs={
            "training": step_process.properties.ProcessingOutputConfig
                .Outputs["tfrecords"].S3Output.S3Uri
        },
        cache_config=cache_cfg,
    )

    # ----------------------------
    # Step 4: Create SageMaker Model
    # ----------------------------
    
    model = TensorFlowModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        framework_version="2.13",
        sagemaker_session=sm_pipe_sess,
    )
    
    step_create_model = ModelStep(
        name="CreateModel",
        step_args=model.create(
            instance_type="ml.m5.large",
        ),
    )

    # ----------------------------
    # Step 5: Deploy Model via Lambda
    # ----------------------------

    current_time = generate_uuid()
    deploy_model_lambda_function_name = f"sagemaker-deploy-model-lambda-{current_time}"
    
    deploy_model_lambda = Lambda(
        function_name=deploy_model_lambda_function_name,
        execution_role_arn=_resolve_lambda_role(),
        script="deploy_model_lambda.py",
        handler="deploy_model_lambda.lambda_handler",
        session=sagemaker.Session(boto_session=boto_sess),
    )

    step_deploy_lambda = LambdaStep(
        name="DeployModel",
        lambda_func=deploy_model_lambda,
        inputs={
            "model_name": step_create_model.properties.ModelName,
            "endpoint_instance_type": "ml.m5.large",
            "endpoint_config_name": "cheese-app-endpoint-config",
            "endpoint_name": "cheese-app-endpoint"
        },
        cache_config=cache_cfg,
    )

    # ---------------
    # Build & execute
    # ---------------
    sm_pipeline = Pipeline(
        name="cheese-pipeline",
        steps=[step_collect, step_process, step_train, step_create_model, step_deploy_lambda],
        sagemaker_session=sm_pipe_sess,
    )

    sm_pipeline.upsert(role_arn=role)
    execution = sm_pipeline.start()
    print(f"SageMaker Pipeline started: {execution.arn}")


def main(args=None):
    print("CLI Arguments:", args)

    if args.data_collector:
        data_collector()

    if args.data_processor:
        print("Data Processor")
        data_processor()

    if args.model_training:
        print("Model Training")
        model_training()

    if args.model_deploy:
        print("Model Deploy")
        model_deploy()

    if args.pipeline:
        pipeline()

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "--data_collector",
        action="store_true",
        help="Run just the Data Collector",
    )
    parser.add_argument(
        "--data_processor",
        action="store_true",
        help="Run just the Data Processor",
    )
    parser.add_argument(
        "--model_training",
        action="store_true",
        help="Run just Model Training",
    )
    parser.add_argument(
        "--model_deploy",
        action="store_true",
        help="Run just Model Deployment",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Cheese App Pipeline",
    )

    args = parser.parse_args()

    main(args)
