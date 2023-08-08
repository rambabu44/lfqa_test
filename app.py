from haystack.document_stores import PineconeDocumentStore

document_store = PineconeDocumentStore(
    api_key='3e182c06-0421-40b6-b291-4566b9026352',
    index='lfqa',
    similarity="cosine",
    embedding_dim=768,
    environment="us-west1-gcp-free"
)

from haystack import Document

docs = []
with open('questions.txt', 'r') as fileq:
    with open('answers.txt', 'r') as filea:
        while True:
            q=fileq.readline()
            a=filea.readline()
            if not q or not a:
                break
            doc = Document(
                content=a,
                meta={
                    "question": q
                }
            )
            print(a)
            docs.append(doc)

document_store.write_documents(docs)
print(document_store.get_document_count())
from haystack.nodes import EmbeddingRetriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers"
)
document_store.update_embeddings(
    retriever,
    batch_size=16
)
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser

rag_prompt = PromptTemplate(
    prompt="""Synthesize a comprehensive answer from the following text for the given question.
                             Provide a clear and concise response that summarizes the key points and information presented in the text.
                             Your answer should be in your own words and be no longer than 50 words.
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(model_name_or_path="google/flan-t5-large", default_prompt_template=rag_prompt)

from haystack.pipelines import Pipeline

pipe = Pipeline()
pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

output = pipe.run(query="Can i claim for suicidal case?")

print(output["answers"][0].answer)

# from haystack.nodes import Seq2SeqGenerator
# generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")


# from haystack.nodes import gen
# generator = RAGenerator(
#     model_name_or_path="facebook/rag-token-nq",
#     use_gpu=True,
#     top_k=1,
#     max_length=200,
#     min_length=2,
#     embed_title=True,
#     num_beams=2,
# )


# from haystack.pipelines import GenerativeQAPipeline

# pipe = GenerativeQAPipeline(generator, retriever)

# result = pipe.run(
#         query="What is claim?",
#         params={
#             "Retriever": {"top_k": 3},
#             "Generator": {"top_k": 1}
#         })

# print(result)



