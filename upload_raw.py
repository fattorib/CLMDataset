import boto3
from keys import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY

if __name__ == "__main__":
    PATH = "data/interim/train/webtextjsonl.tgz"
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    with open(PATH, "rb") as f:
        s3.upload_fileobj(f, "openwebtxtbf", f"processed/openwebtxtcleaned.tgz")
