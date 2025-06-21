from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

