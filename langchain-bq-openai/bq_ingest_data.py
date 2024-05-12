from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import BigQueryVectorSearch
from langchain.document_loaders import GCSFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'filename.json'

PROJECT_ID = "project-id"

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=PROJECT_ID
)
gcs_bucket_name = "bucket-name"
pdf_filename = "test_data/example.pdf"

def load_pdf(file_path):
    return PyPDFLoader(file_path)

loader = GCSFileLoader(
    project_name=PROJECT_ID, bucket=gcs_bucket_name, blob=pdf_filename, loader_func=load_pdf
)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)
doc_splits = text_splitter.split_documents(documents)

for idx, split in enumerate(doc_splits):
    split.metadata["chunk"] = idx

print(f"# of documents = {len(doc_splits)}")

DATASET = "bq_vectordb"
TABLE = "bq_vectors"

bq_object = BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location="US",
    embedding=embedding,
)

bq_object.add_documents(doc_splits)