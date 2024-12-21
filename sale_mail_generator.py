import streamlit as st
import textwrap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re


def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = re.sub(r'\s{2,}', ' ', text).strip()  # Normalize whitespace
    return text


# Portfolio management class
class Portfolio:
    def __init__(self, file_path="my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.vectorizer = TfidfVectorizer()
        self.index = None
        self.vectors = None

    def load_portfolio(self):
        tech_stacks = self.data["Techstack"].values
        self.vectors = self.vectorizer.fit_transform(tech_stacks).toarray()
        dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.vectors).astype('float32'))

    def query_links(self, skills):
        if isinstance(skills, list):
            skills = " ".join(skills)
        query_vector = self.vectorizer.transform([skills]).toarray().astype('float32')
        _, indices = self.index.search(query_vector, k=2)
        results = [{"links": self.data.iloc[idx]["Links"]} for idx in indices[0]]
        return results


# Chain management class for LLM interactions
class Chain:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyC2dXe_sPE3QfEdkFQrlp3s-S2dduAGRTA")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        json_parser = JsonOutputParser()
        return json_parser.parse(res.content)


    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            ### INSTRUCTION:
            You are BDE at FAST OFFER, FAST OFFER is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of FAST OFFER 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase FAST OFFER's portfolio: {link_list}
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content


# Streamlit app functions
def set_custom_css():
    st.markdown(
        """
        <style>
        /* Import Poppins font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        /* Apply Poppins font to all elements */
        html, body, div, span, applet, object, iframe,
        h1, h2, h3, h4, h5, h6, p, blockquote, pre,
        a, abbr, acronym, address, big, cite, code,
        del, dfn, em, img, ins, kbd, q, s, samp,
        small, strike, strong, sub, sup, tt, var,
        b, u, i, center, dl, dt, dd, ol, ul, li,
        fieldset, form, label, legend, table, caption,
        tbody, tfoot, thead, tr, th, td, article,
        aside, canvas, details, embed, figure,
        figcaption, footer, header, hgroup, menu,
        nav, output, ruby, section, summary, time,
        mark, audio, video, textarea {
            font-family: 'Poppins', sans-serif;
        }

        /* Customize the text areas */
        textarea {
            font-size: 16px;
        }

        /* Customize the color and appearance of markdown headings */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )





def display_links():
    st.markdown(
        """
        <div style="border: 1px dashed #9d4edd; padding: 10px; border-radius: 8px; font-family: Poppins;">
        <strong>💡 Note:</strong> Ensure the URLs you provide are valid and contain accurate job descriptions.<br>
        <strong>Example Links:</strong><br>
        https://japan-dev.com/jobs/geniee/geniee-aigenerative-ai-engineer-ixsc0h<br>
        https://japan-dev.com/jobs/rakuten/rakuten-project-manager---technology-platforms-office-6fals7<br>
        </div>
        <br>
        """,
        unsafe_allow_html=True
    )


def create_streamlit_app(chain, portfolio):
    st.set_page_config(layout="wide", page_title="Cold mail Generator", page_icon="📧")
    set_custom_css()
    st.markdown("<h1 style='color: #FFFFFF;'> 🌟SALES MAIL GENERATOR</h1>", unsafe_allow_html=True)
    st.write("""
    Sales mail generator is AI tool to generate professional sales mails.
     
    ➡️ Follow these steps to generate a cold email:
    1. Paste the job description webpage link.
    2. Ensure the link contains correct job details.
    """)

    url_input = st.text_input("Paste web URL here:")
    submit_button = st.button("Generate")
    display_links()

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = chain.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = chain.write_mail(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


# Main execution
if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    create_streamlit_app(chain, portfolio)