rm -f trainer.tar trainer.tar.gz
tar cvf trainer.tar package
gzip trainer.tar
aws s3 cp trainer.tar.gz $S3_PACKAGE_URI/cheese-app-trainer.tar.gz
