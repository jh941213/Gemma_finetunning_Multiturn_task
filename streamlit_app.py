from dataclasses import dataclass
from typing import Literal
import streamlit as st
from streamlit import sidebar
import fitz  # PyMuPDF
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components
from langchain.prompts.prompt import PromptTemplate

openai_api_key= "your_openai_api_key_here"



sys_prompt = """
### 지시사항 :
당신은 디노랩스 발표회에서 시연을 하게 될 챗봇입니다. 친근한 말투로 대화를 나누어주세요.

### 디노랩스  
디노랩스는 인공지능 교육 기업입니다.
디노랩스는 '알디노'가 만든 AI 스터디 커뮤니티로 집단지성을 추구하는 모임입니다. 최대 주주는 DINO LABS입니다.
디노랩스의 CEO는 이원재입니다. 디노랩스의 운영진은 황성연, 이현중 입니다.

### Current conversation:
{history}

### Question : {input}
Answer :
"""


PROMPT = PromptTemplate(input_variables=["history", "input"], template=sys_prompt)


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str
def handle_pdf_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            # Here you can process the PDF file as needed
            # For example, extracting text from the first page:
            first_page_text = doc[0].get_text()
            st.sidebar.text_area("PDF Content Preview:", first_page_text, height=300)
        except Exception as e:
            st.sidebar.write("Error handling PDF:", e)


def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo-0125"  # 이 부분은 최신 모델로 업데이트가 필요할 수 있습니다.
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm)
        )

def on_click_callback():
    # 대화 이력을 문자열로 변환
    history_str = "\n".join([f"{msg.origin}: {msg.message}" for msg in st.session_state.history])
    # 사용자 입력을 받음
    user_input = st.session_state.human_prompt
    # PromptTemplate을 사용하여 현재 대화 상태와 사용자 입력을 포맷
    formatted_prompt = PROMPT.format(history=history_str, input=user_input)  # 여기를 수정함

    with get_openai_callback() as cb:
        # 포맷된 프롬프트를 사용하여 언어 모델의 응답을 얻음
        llm_response = st.session_state.conversation.invoke({'input':formatted_prompt})
        response_text = llm_response['response'] if 'response' in llm_response else ''
        
        response_text += "<br>"
        
        response_text += "🤖 저는 디노랩스에서 가스라이팅 당한 챗봇입니다."

        st.session_state.history.append(
            Message("human", user_input)
        )
        st.session_state.history.append(
            Message("ai", response_text)
        )
        st.session_state.token_count += cb.total_tokens



load_css()
initialize_session_state()

with st.sidebar:
    st.title("Upload your data")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    handle_pdf_upload(uploaded_file)


st.title("DINO LABS Gemma-Ko chatbot 🤖")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row 
    {'' if chat.origin == 'ai' else 'row-reverse'}">
    <img class="chat-icon" src="app/static/{
        'dino.png' if chat.origin == 'ai' 
                      else 'holland.png'}"
         width=32 height=32>
    <div class="chat-bubble
    {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
        &#8203;{chat.message}
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)
    
    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**📝 텍스트를 입력하세요.**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat with WIZnet Chatbot 🤖",
        placeholder="디노랩스는 무엇을 하는 곳이야?",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="primary", 
        on_click=on_click_callback, 
    )

credit_card_placeholder.caption(f"""
Used {st.session_state.token_count} tokens \n
대화내역: 
{st.session_state.conversation.memory.buffer}
""")

components.html("""
<script>
const streamlitDoc = window.parent.document;

const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);

streamlitDoc.addEventListener('keydown', function(e) {
    switch (e.key) {
        case 'Enter':
            submitButton.click();
            break;
    }
});
</script>
""", 
    height=0,
    width=0,
)
