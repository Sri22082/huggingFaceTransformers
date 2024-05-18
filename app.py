import streamlit as st
from transformers import pipeline, TFAutoModelForQuestionAnswering, AutoTokenizer

@st.cache_resource
def load_qa_model():
    model_name = "distilbert-base-cased-distilled-squad"
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, framework='tf')
    return qa_pipeline

qa = load_qa_model()

st.title("Ask Questions about Your Text")
st.write("Paste a passage and ask a question about it.")

sentence = st.text_area('Please paste your article:', height=300)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")

max_length = st.sidebar.slider('Select max answer length', 50, 500, step=10, value=150)
min_length = st.sidebar.slider('Select min answer length', 10, 450, step=10, value=50)
do_sample = st.sidebar.checkbox("Do sampling", value=False)

with st.spinner("Discovering Answers.."):
    if button and sentence and question:
        answers = qa(question=question, context=sentence, max_answer_len=max_length, min_answer_len=min_length, do_sample=do_sample)
        st.write("Answer:", answers['answer'])


# streamlit run app.py
