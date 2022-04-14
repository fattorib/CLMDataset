import boto3
from keys import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY

if __name__ == "__main__":
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    s3.download_file(
        "openwebtxtbf", "processed/openwebtext_cleaned.tar.gz", "data/interim/train/corpus.tar.gz"
    )
