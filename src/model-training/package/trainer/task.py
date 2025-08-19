import argparse
import os
import time
import zipfile

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K

import boto3

# Setup the arguments for the trainer task
parser = argparse.ArgumentParser()
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument(
    "--model_name",
    dest="model_name",
    default="mobilenetv2",
    type=str,
    help="Model name",
)
parser.add_argument(
    "--train_base",
    dest="train_base",
    default=False,
    action="store_true",
    help="Train base or not",
)
parser.add_argument(
    "--epochs", dest="epochs", default=10, type=int, help="Number of epochs."
)
parser.add_argument(
    "--batch_size", dest="batch_size", default=16, type=int, help="Size of a batch."
)
parser.add_argument(
    "--bucket_name",
    dest="bucket_name",
    default="",
    type=str,
    help="Bucket for data and models.",
)
parser.add_argument(
    "--model_dir",
    dest="model_dir",
    default="",
    type=str,
    help="Model directory (provided by SageMaker).",
)
args = parser.parse_args()

# TF Version
print("tensorflow version", tf.__version__)
print("Eager Execution Enabled:", tf.executing_eagerly())
# Get the number of replicas
strategy = tf.distribute.MirroredStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

devices = tf.config.experimental.get_visible_devices()
print("Devices:", devices)
print(tf.config.experimental.list_logical_devices("GPU"))

print("GPU Available: ", tf.config.list_physical_devices("GPU"))
print("All Physical Devices", tf.config.list_physical_devices())

num_classes = 3

# Determine data path based on environment (SageMaker vs local)
if os.path.exists('/opt/ml'):
    # Running in SageMaker - input data is in /opt/ml/input/data/training/
    sagemaker_input_path = "/opt/ml/input/data/training"
    dataset_folder = "/opt/ml/input/data/training"
    
    # Check if data was provided as zip file and extract
    zip_file_path = os.path.join(sagemaker_input_path, "tfrecords.zip")
    if os.path.exists(zip_file_path):
        print(f"Found tfrecords.zip at {zip_file_path}, extracting...")
        with zipfile.ZipFile(zip_file_path) as zfile:
            zfile.extractall(dataset_folder)
        tfrecords_folder = os.path.join(dataset_folder, "tfrecords")
    else:
        # Assume tfrecords are directly in the input folder
        tfrecords_folder = sagemaker_input_path
        print(f"Using tfrecords directly from {tfrecords_folder}")
else:
    # Running locally
    dataset_folder = os.path.join("/persistent", "dataset")
    tfrecords_folder = os.path.join(dataset_folder, "tfrecords")
    
    # Make dirs
    os.makedirs(dataset_folder, exist_ok=True)

    # Check if tfrecords exists locally
    if not os.path.exists(tfrecords_folder):
        print(f"TFRecords not found at {tfrecords_folder}, downloading from S3...")
        # Download Data
        start_time = time.time()
        s3_client = boto3.client('s3')
        source_object_key = "tfrecords.zip"
        destination_file_name = os.path.join(dataset_folder, "tfrecords.zip")
        s3_client.download_file(args.bucket_name, source_object_key, destination_file_name)

        # Unzip data
        with zipfile.ZipFile(destination_file_name) as zfile:
            zfile.extractall(dataset_folder)
        execution_time = (time.time() - start_time) / 60.0
        print("Download execution time (mins)", execution_time)
    else:
        print(f"TFRecords found at {tfrecords_folder}")

print(f"Using tfrecords from: {tfrecords_folder}")


# Create TF Datasets using TF records
def get_dataset_tfrecord(
    image_width=224, image_height=224, num_channels=3, batch_size=32
):
    # Read TF Records
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    # @tf.function
    def parse_tfrecord_example(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)

        # Image
        # image = tf.image.decode_jpeg(parsed_example['image'])
        image = tf.io.decode_raw(parsed_example["image"], tf.uint8)
        image.set_shape([num_channels * image_height * image_width])
        image = tf.reshape(image, [image_height, image_width, num_channels])
        # Label
        label = tf.cast(parsed_example["label"], tf.int32)
        label = tf.one_hot(label, num_classes)

        return image, label

    # Normalize pixels
    def normalize(image, label):
        image = image / 255
        return image, label

    # Read the tfrecord files
    train_tfrecord_files = tf.data.Dataset.list_files(tfrecords_folder + "/train*")
    validate_tfrecord_files = tf.data.Dataset.list_files(tfrecords_folder + "/val*")

    #############
    # Train data
    #############
    train_data = train_tfrecord_files.flat_map(tf.data.TFRecordDataset)
    train_data = train_data.map(
        parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_data = train_data.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    ##################
    # Validation data
    ##################
    validation_data = validate_tfrecord_files.flat_map(tf.data.TFRecordDataset)
    validation_data = validation_data.map(
        parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE
    )
    validation_data = validation_data.map(
        normalize, num_parallel_calls=tf.data.AUTOTUNE
    )
    validation_data = validation_data.batch(batch_size)
    validation_data = validation_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, validation_data


def build_mobilenet_model(
    image_height, image_width, num_channels, num_classes, model_name, train_base=False
):
    # Model input
    input_shape = [image_height, image_width, num_channels]  # height, width, channels

    # Load a pretrained model from keras.applications
    tranfer_model_base = keras.applications.MobileNetV2(
        input_shape=input_shape, weights="imagenet", include_top=False
    )

    # Freeze the mobileNet model layers
    tranfer_model_base.trainable = train_base

    # Regularize using L1
    kernel_weight = 0.02
    bias_weight = 0.02

    model = Sequential(
        [
            tranfer_model_base,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(
                units=128,
                activation="relu",
                kernel_regularizer=keras.regularizers.l1(kernel_weight),
                bias_regularizer=keras.regularizers.l1(bias_weight),
            ),
            keras.layers.Dense(
                units=num_classes,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l1(kernel_weight),
                bias_regularizer=keras.regularizers.l1(bias_weight),
            ),
        ],
        name=model_name + "_train_base_" + str(train_base),
    )

    return model


print("Start Train model")
############################
# Training Params
############################
model_name = args.model_name
learning_rate = 0.001
image_width = 224
image_height = 224
num_channels = 3
batch_size = args.batch_size
epochs = args.epochs
train_base = args.train_base

# Free up memory
K.clear_session()

# Data
train_data, validation_data = get_dataset_tfrecord(
    image_width=image_width,
    image_height=image_height,
    num_channels=num_channels,
    batch_size=batch_size,
)

if model_name == "mobilenetv2":
    # Model
    model = build_mobilenet_model(
        image_height,
        image_width,
        num_channels,
        num_classes,
        model_name,
        train_base=train_base,
    )
    # Optimizer
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    # Loss
    loss = keras.losses.categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])


# Train model
start_time = time.time()
training_results = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    verbose=1,
)
execution_time = (time.time() - start_time) / 60.0
print("Training execution time (mins)", execution_time)

print("Change model signature and save")


# Preprocess Image
def preprocess_image(bytes_input):
    decoded = tf.io.decode_jpeg(bytes_input, channels=3)
    decoded = tf.image.convert_image_dtype(decoded, tf.float32)
    resized = tf.image.resize(decoded, size=(224, 224))
    return resized


@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def preprocess_function(bytes_inputs):
    decoded_images = tf.map_fn(
        preprocess_image, bytes_inputs, dtype=tf.float32, back_prop=False
    )
    return {"model_input": decoded_images}


@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def serving_function(bytes_inputs):
    images = preprocess_function(bytes_inputs)
    results = model_call(**images)
    return results


model_call = tf.function(model.call).get_concrete_function(
    [tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="model_input")]
)

# Save as a TF Serving bundle under a versioned directory so SageMaker can find it
if os.path.exists('/opt/ml'):
    # SageMaker training environment: prefer SM_MODEL_DIR if present
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    ARTIFACT_URI = os.path.join(sm_model_dir, "1")  # version subfolder required
else:
    # Local environment
    ARTIFACT_URI = os.path.join("/persistent", "model", "1")

os.makedirs(ARTIFACT_URI, exist_ok=True)

tf.saved_model.save(
    model,
    ARTIFACT_URI,
    signatures={"serving_default": serving_function},
)

print(f"Model saved to: {ARTIFACT_URI}")

print("Training Job Complete")
