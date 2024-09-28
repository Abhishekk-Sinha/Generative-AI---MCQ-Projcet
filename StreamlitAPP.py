import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain  # Back to using LLMChain
from langchain.chains import SequentialChain  # Use SequentialChain if RunnableSequence is not available
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# Load environment variables from the .env file
load_dotenv()

# Load JSON file
with open('D:/Generative AI - MCQ Projcet/Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Initialize the OpenAI model
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo", temperature=0.7)

# Define the quiz generation prompt
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template="""
    Text: {text}
    You are an expert MCQ maker. Given the above text, create {number} multiple choice questions for {subject} students in {tone} tone.
    Format your response like the RESPONSE_JSON:
    {response_json}
    """
)

# Define the quiz evaluation prompt
quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template="""
    You are an expert English grammarian. Evaluate the quiz below for {subject} students:
    {quiz}
    """
)

# Define the quiz chain using LLMChain
quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz")
review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review")

# Combine chains using SequentialChain
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True
)

# Streamlit UI
st.title("MCQs Creator Application with LangChain")

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or txt file", type=["pdf", "txt"])
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Insert Subject", max_chars=20)
    tone = st.text_input("Complexity Level of Question", max_chars=20, placeholder="Simple")
    button = st.form_submit_button("Create MCQs")

if button and uploaded_file and mcq_count and subject and tone:
    with st.spinner("Loading..."):
        try:
            # Convert uploaded file to text
            text = read_file(uploaded_file)

            # Generate MCQs and evaluate them
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                )

                st.write(f"Total Tokens: {cb.total_tokens}")
                st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                st.write(f"Completion Tokens: {cb.completion_tokens}")
                st.write(f"Total Cost (USD): ${cb.total_cost}")

            # Extract quiz and review
            if isinstance(response, dict):
                quiz = response.get("quiz", None)
                if quiz:
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        st.table(df)
                        st.text_area(label="Review", value=response["review"])
                    else:
                        st.error("Error in the table data.")
                else:
                    st.error("Error in the quiz data.")
            else:
                st.error("Error in the response format.")

        except Exception as e:
            st.error("An error occurred while processing.")
            st.error(traceback.format_exc())
