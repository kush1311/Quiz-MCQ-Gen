import streamlit as st
import json
import PyPDF2
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# Initialize the LLM with your API key
KEY = "ChatGPT-API-KEY"
llm = ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=0.5)

# Define the JSON template for quiz generation
RESPONSE_JSON = {
    "1": {"mcq": "multiple choice question", "options": {"a": "choice here", "b": "choice here", "c": "choice here", "d": "choice here"}, "correct": "correct answer"},
    "2": {"mcq": "multiple choice question", "options": {"a": "choice here", "b": "choice here", "c": "choice here", "d": "choice here"}, "correct": "correct answer"},
    "3": {"mcq": "multiple choice question", "options": {"a": "choice here", "b": "choice here", "c": "choice here", "d": "choice here"}, "correct": "correct answer"},
}

# Prompt template for quiz generation
TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, create a quiz of {number} multiple choice questions for {subject} students with {difficulty} difficulty. 
Make sure the questions are not repeated and format your response like RESPONSE_JSON below. Use it as a guide and ensure to create {number} MCQs.
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(input_variables=["text", "number", "subject", "difficulty", "response_json"], template=TEMPLATE)

# Chain to generate quiz
quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

# Streamlit UI
st.title("Quiz Generator")

# User inputs
uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])
NUMBER = st.number_input("Number of MCQs", min_value=1, max_value=10, value=5)
SUBJECT = st.text_input("Subject", value="Biology")
DIFFICULTY = st.selectbox("Difficulty Level", ["easy", "medium", "hard"])

TEXT = ""
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            TEXT += page.extract_text()
    elif uploaded_file.type == "text/plain":
        TEXT = uploaded_file.read().decode("utf-8")

    if st.button("Generate Quiz"):
        response = quiz_chain({
            "text": TEXT,
            "number": NUMBER,
            "subject": SUBJECT,
            "difficulty": DIFFICULTY,
            "response_json": json.dumps(RESPONSE_JSON)
        })

        quiz = response.get("quiz")
        quiz = json.loads(quiz)

        st.subheader("Generated Quiz")

        for key, value in quiz.items():
            st.write(f"**Q{key}: {value['mcq']}**")
            for option, option_value in value["options"].items():
                st.write(f"{option}: {option_value}")
            st.write(f"**Correct Answer: {value['correct']}**")
            st.markdown("---")
