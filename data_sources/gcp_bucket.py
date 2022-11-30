from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
