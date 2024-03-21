import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import FAISS
from flask import Flask, render_template, request, jsonify
from langchain import PromptTemplate
from langchain import HuggingFacePipeline
app = Flask(__name__)

# Loading embeddings model
model_name = 'hkunlp/instructor-base'
embedding_model = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs={"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
)

# Loading GPT-2 model
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    task="text-generation",  # Use text-generation task for language modeling
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 100,
    max_length = 100
)

llm = HuggingFacePipeline(pipeline=pipe)

# Setting up conversational retrieval chain
question_generator = LLMChain(
    llm=llm,
    prompt=CONDENSE_QUESTION_PROMPT,
    verbose=True
)

prompt_template = """
**Welcome to AIT-GPT!**  I'm your friendly AI assistant for all things AIT.

**Ask me anything related to the Asian Institute of Technology:** Whether you're curious about academic programs, research areas, faculty expertise, or campus life, I'm here to help!

**For the best results, try using specific keywords or phrases** related to your question.

**Here are some examples to get you started:**

* What are the admission requirements for international students?
* What are the current research areas at AIT?
* Can you tell me more about the faculty in the School of Engineering and Technology?
* When was AIT founded?
* What are some of the student clubs and organizations available at AIT?

**Now, ask your question!**
**Context: {context}**
**Question: {question}**

**Answer:**
    """.strip()
PROMPT = PromptTemplate.from_template(template=prompt_template)

vector_path = './vector-store'
vectordb = FAISS.load_local(
    folder_path=vector_path,
    embeddings=embedding_model,
    index_name="nlp"
)

retriever = vectordb.as_retriever()  # Your retriever object here

doc_chain = load_qa_chain(
    llm=llm,
    chain_type='stuff',
    prompt=PROMPT,
    verbose=True
)

memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h: h
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    print("User message:   ",user_message)
    bot_response = chain({"question":user_message})
    print("AIT-GPT is response:   ",bot_response['answer'])
    return jsonify({'message': bot_response['answer']})

if __name__ == '__main__':
    app.run(debug=True)
