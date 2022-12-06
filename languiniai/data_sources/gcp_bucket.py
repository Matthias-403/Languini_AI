from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('languini-ai-bucket')

# blob = f'{file_path_in_bucket}/{file_name}'
# https://cloud.google.com/appengine/docs/legacy/standard/python/googlecloudstorageclient/read-write-to-cloud-storage#reading_from
def list_files(prefix: str = '') -> list:
    '''
    input (prefix): directory to be listed
    List files in a given prefix(directory in bucket)
    '''
    return list(map(lambda blob: blob.name, bucket.list_blobs(prefix=prefix)))


def load_files(files_name: list = [], read_byte: bool = True) -> list:
    '''
    input (files_name): list of full_name (<path>/<file_name>) in the bucket
    input (read_byte): True to read file in non-text encoded format
    output: list of file read into memory
    Read a blob from GCS using file-like IO
    '''
    read_type = 'r'
    if read_byte: read_type = read_type + 'b'
    def read_file(blob):
        b = bucket.blob(blob)
        with b.open(read_byte) as f:
            return f.read()
    
    if files_name:
        return list(map(read_file, files_name))


