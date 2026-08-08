import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.text_splitter import TokenTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def get_config(key: str) -> str | None:
    """Read config from Streamlit Cloud secrets first, fall back to local .env."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # st.secrets raises if no secrets.toml exists locally — that's fine
    return os.getenv(key)


st.set_page_config(page_title="Malak's AI Assistant", page_icon="🤖", layout="centered")

st.markdown("""
<style>

/* ==========================================================
   IMPORT FONT
========================================================== */

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');


/* ==========================================================
   GLOBAL BACKGROUND
========================================================== */

html,
body,
[data-testid="stAppViewContainer"],
.stApp{

    background-color:#050816 !important;

    background-image:
        linear-gradient(rgba(99,102,241,.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,.12) 1px, transparent 1px);

    background-size:40px 40px;

    background-attachment:fixed;

    color:white;

}


/* ==========================================================
   BLUE GLOW
========================================================== */

.stApp::before{

    content:"";

    position:fixed;

    inset:0;

    background:
        radial-gradient(circle at top,
        rgba(59,130,246,.20),
        transparent 55%);

    pointer-events:none;

    z-index:0;

}


/* ==========================================================
   MAIN CONTAINER
========================================================== */

.main .block-container{

    position:relative;

    z-index:2;

    padding-top:3rem;

    background:transparent;

}


/* ==========================================================
   TITLE
========================================================== */

h1{

    font-family:'Orbitron',sans-serif !important;

    font-size:3rem !important;

    font-weight:700 !important;

    background:linear-gradient(
        90deg,
        #8b5cf6,
        #6366f1,
        #3b82f6,
        #06b6d4
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

    text-shadow:
        0 0 25px rgba(99,102,241,.45);

}


/* ==========================================================
   CAPTION
========================================================== */

p{

    color:#9ca3af;

    font-family:'Inter',sans-serif;

}


/* ==========================================================
   CHAT CARDS
========================================================== */

[data-testid="stChatMessage"]{

    background:rgba(20,20,30,.78);

    backdrop-filter:blur(18px);

    border-radius:22px;

    border:1px solid rgba(99,102,241,.20);

    box-shadow:

        0 0 25px rgba(59,130,246,.10),

        inset 0 0 8px rgba(255,255,255,.03);

    padding:18px;

    transition:.25s;

}

[data-testid="stChatMessage"]:hover{

    transform:translateY(-2px);

    border:1px solid rgba(59,130,246,.45);

    box-shadow:

        0 0 35px rgba(59,130,246,.20);

}


/* ==========================================================
   CHAT INPUT
========================================================== */

[data-testid="stChatInput"]{

    background:rgba(18,18,30,.95);

    border-radius:20px;

    border:1px solid rgba(99,102,241,.35);

    box-shadow:

        0 0 25px rgba(59,130,246,.18);

}


/* ==========================================================
   INPUT TEXT
========================================================== */

textarea{

    color:white !important;

}


/* ==========================================================
   SEND BUTTON
========================================================== */

button{

    border-radius:18px !important;

    transition:.25s;

}

button:hover{

    box-shadow:

        0 0 20px rgba(59,130,246,.45);

}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"]{

    background:#0a0f1f;

    border-right:1px solid rgba(99,102,241,.20);

}


/* ==========================================================
   SCROLLBAR
========================================================== */

::-webkit-scrollbar{

    width:10px;

}

::-webkit-scrollbar-track{

    background:#050816;

}

::-webkit-scrollbar-thumb{

    background:#3b82f6;

    border-radius:20px;

}


/* ==========================================================
   REMOVE STREAMLIT HEADER
========================================================== */

header{

    background:transparent !important;

}

footer{

    visibility:hidden;

}

</style>
""", unsafe_allow_html=True)

st.title("🤖 Malak's Personal AI Assistant")
st.caption("Ask me anything about Malak's background, skills, or projects.")
# ============================================================
# END DESIGN BLOCK
# ============================================================


@st.cache_resource(show_spinner="Loading knowledge base...")
def build_chain():
    llm = ChatOpenAI(
        api_key=get_config("OPENROUTER_API_KEY"),
        base_url=get_config("OPENROUTER_BASE_URL"),
        model=get_config("OPENROUTER_MODEL"),
        temperature=0.0,
    )

    loader = TextLoader("Data.txt", encoding="utf-8")
    documents = loader.load()

    splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=30)
    splits = splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_db = FAISS.from_documents(splits, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are my personal AI assistant.

You answer questions about me using ONLY the provided context.

If the answer is not found in the context, reply:
"I don't have that information, Contact Malak Directly for more information."

Context:
{context}
""",
            ),
            ("human", "{question}"),
        ]
    )

    return llm, retriever, prompt


llm, retriever, prompt = build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)

            messages = prompt.format_messages(context=context, question=question)

            answer = None
            for attempt in range(3):
                response = llm.invoke(messages)
                if response.content:
                    answer = response.content
                    break
            if not answer:
                answer = "I'm having trouble generating a response right now — please try asking again."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
