import os
import json
import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2
from langchain.chat_models import ChatOpenAI


KEY=("OPENAI_API_KEY")
llm=ChatOpenAI(openai_api_key=KEY,model_name="gpt-3.5-turbo", temperature=0.5)

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""


quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
    )


quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""


quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE)
review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)
generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)
file_path=r"File Path"




with open(file_path, 'r') as file:
    TEXT = file.read()
print(TEXT)

json.dumps(RESPONSE_JSON)

NUMBER=5 
SUBJECT="biology"
TONE="simple"


with get_openai_callback() as cb:
    response=generate_evaluate_chain(
        {
            "text": TEXT,
            "number": NUMBER,
            "subject":SUBJECT,
            "tone": TONE,
            "response_json": json.dumps(RESPONSE_JSON)
        }
        )

quiz=response.get("quiz")

quiz=json.loads(quiz)
quiz_table_data = []
for key, value in quiz.items():
    mcq = value["mcq"]
    options = " | ".join(
        [
            f"{option}: {option_value}"
            for option, option_value in value["options"].items()
            ]
        )
    correct = value["correct"]
    quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

quiz=pd.DataFrame(quiz_table_data)

quiz.to_csv("machinelearning.csv",index=False)
