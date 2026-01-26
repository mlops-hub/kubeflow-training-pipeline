import boto3
import tempfile
import joblib

def parse_s3_uri(uri: str):
    assert uri.startswith("s3://")
    _, bucket, *key_parts = uri.split("/")
    key = "/".join(key_parts)
    return bucket, key

def load_model_from_uri(uri: str):
    bucket, key = parse_s3_uri(uri)
    print('bucket: ', bucket)
    print('key: ', key)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, tmp.name)

    return joblib.load(tmp.name)