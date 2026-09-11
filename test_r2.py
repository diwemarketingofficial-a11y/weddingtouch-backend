from dotenv import load_dotenv

load_dotenv()

from r2_storage import upload_file, create_download_url

data = b"Hello Wedding Touch R2!"

key = upload_file(
    data,
    "test/hello.txt",
    "text/plain",
)

print("Uploaded successfully!")
print("R2 key:", key)

url = create_download_url(key)

print("Temporary URL:")
print(url)