from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="deepseek-r1")

template = """
You are a helpful assistant. You are an expert in answering questions about a world population.
Here are some relevant stats: {reviews}
Here is a question: {question}
Please provide a concise and informative answer. If it is a simple question, provide a simple one line answer. If it is a complex question, provide a detailed answer.
Avoid unnecessary information and focus on the question asked. Avoid repeating the question in your answer.
Use figures and statistics from the provided data to support your answer."""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    question = input(
        "Enter a question about the Kid's cycle (or 'q' to quit): ")
    if question.lower() == 'q':
        break

    reviews = retriever.invoke(question)

    result = chain.invoke(
        {"reviews": reviews, "question": question})
    print(result)
