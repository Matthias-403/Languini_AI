from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('languini-ai-bucket')

# blob = f'{file_path_in_bucket}/{file_name}'
# https://cloud.google.com/appengine/docs/legacy/standard/python/googlecloudstorageclient/read-write-to-cloud-storage#reading_from
def load_files(files_name: list = []) -> list:
    """Read a blob from GCS using file-like IO"""

    def read_file(blob):
        b = bucket.blob(blob)
        with b.open("rb") as f:
            return f.read()
    
    if files_name:
        return list(map(read_file, files_name))

def list_files(prefix: str = ''):
    """List files in a given prefix(directory in bucket)"""
    return list(map(lambda blob: blob.name, bucket.list_blobs(prefix=prefix)))
