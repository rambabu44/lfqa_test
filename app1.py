from haystack import Document
from haystack.document_stores import PineconeDocumentStore

document_store = PineconeDocumentStore(
    api_key='3e182c06-0421-40b6-b291-4566b9026352',
    index='lfqa',
    similarity="cosine",
    embedding_dim=768,
    environment="us-west1-gcp-free"
)
docs = []
with open('questions.txt', 'r') as fileq:
    q=fileq.readlines()
with open('answers.txt', 'r') as filea:    
    a=filea.readlines()
doc = Document(
    content=a,
    meta={
        "questions": q
    }
)
docs.append(doc)
print(docs)
document_store.write_documents(docs)
print(document_store.get_document_count())
# from haystack.pipelines import DocumentSearchPipeline
# from haystack.utils import print_documents

# from haystack.nodes import EmbeddingRetriever

# retriever = EmbeddingRetriever(
#    document_store=document_store,
#    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
#    model_format="sentence_transformers"
# )

# document_store.update_embeddings(
#    retriever,
#    batch_size=128
# )

# search_pipe = DocumentSearchPipeline(retriever)
# result = search_pipe.run(
#     query="deadline for claim",
#     params={"Retriever": {"top_k": 1}}
# )

# print_documents(result)