# app.py
import os, io, json, uuid
import numpy as np
import streamlit as st
from openai import OpenAI

# -------- OpenAI 예외 클래스(환경별 가드) --------
try:
    from openai import APIError, RateLimitError, AuthenticationError
except Exception:
    APIError = RateLimitError = AuthenticationError = Exception

# -------- PDF 텍스트 추출 --------
try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# ================= 페이지 설정 =================
st.set_page_config(page_title="아이디어 챗봇", page_icon="🐤", layout="wide")

# ================= 전역 스타일 =================
st.markdown("""
<style>
:root{
  --brand:#F4D24B;  /* 노란 포인트 */
  --ink:#222; --ink-weak:#5f6368;
  --bg:#F6F7F9; --card:#fff; --line:rgba(0,0,0,.08); --muted:#EEF0F4;
  --r-xl:24px;
}
html, body, [data-testid="stAppViewContainer"]{
  background:var(--bg);
  color:var(--ink);
  font-family:Pretendard,Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans KR","Malgun Gothic",sans-serif;
}
/* 상단 Share/Star 헤더에 가리지 않게 여백 */
.block-container{ padding-top:6rem !important; padding-bottom:2.5rem !important; }

/* 사이드바 노란 테마 */
[data-testid="stSidebar"]{
  background:var(--brand)!important; border-right:1px solid rgba(0,0,0,.12);
}
[data-testid="stSidebar"] *{ color:#1d1d1d !important; }

/* 사이드바 리스트(히스토리) 라디오를 목록처럼 */
.sidebar-list .stRadio > div{ gap:.5rem; }
.sidebar-list label{ font-weight: 500; }
.sidebar-list [role="radiogroup"]{ display:flex; flex-direction:column; gap:.35rem; }
.sidebar-list [data-baseweb="radio"]{ background:rgba(255,255,255,.55); border:1px solid rgba(0,0,0,.15); border-radius:12px; padding:.35rem .6rem; }
.sidebar-list [aria-checked="true"]{ background:#fff; border-color:rgba(0,0,0,.35); }

/* 카드/배지 */
.card{ background:var(--card); border:1px solid var(--line); border-radius:var(--r-xl);
  box-shadow:0 8px 20px rgba(0,0,0,.05); padding:16px 18px; }
.badge{ display:inline-flex; gap:6px; align-items:center; background:var(--muted);
  border:1px solid var(--line); border-radius:999px; padding:4px 10px; font-size:12px; color:#444; }

/* 말풍선 */
.msg{ display:flex; gap:10px; margin:10px 0; }
.bubble{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:12px 14px; }
.user .bubble{ background:#F8FAFF; border-color:rgba(91,141,239,.35); }
.assistant .bubble{ background:#fff; }
.avatar{ width:30px;height:30px;border-radius:50%; background:var(--muted); display:flex;align-items:center;justify-content:center; }

/* popover에서 버튼 꽉 차게 */
[data-testid="stPopoverContent"] .stButton>button{ width:100%; }

/* 업로더 라벨/빈 줄 제거 & 간격 조정 */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] > div:first-child{ display:none !important; }
[data-testid="stFileUploader"]{ margin-top:-0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ================= 멀티 대화 세션 상태 =================
if "conversations" not in st.session_state:
    st.session_state.conversations = []  # [{"id","title","messages":[{role,content},...]}]
if "active_id" not in st.session_state:
    st.session_state.active_id = str(uuid.uuid4())
    st.session_state.conversations.append({
        "id": st.session_state.active_id, "title": "새 대화", "messages": []
    })

def get_active_convo():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_id:
            return c
    return None

# RAG & 기타 상태
st.session_state.setdefault("rag_ready", False)
st.session_state.setdefault("rag_chunks", [])
st.session_state.setdefault("rag_embeds", None)
st.session_state.setdefault("rag_model", "text-embedding-3-small")
st.session_state.setdefault("use_rag", False)
st.session_state.setdefault("show_upload", False)

# ================= 헬퍼 함수 =================
def ensure_system_message(prompt_text: str):
    convo = get_active_convo()
    msgs = convo["messages"]
    if not msgs or msgs[0].get("role") != "system":
        msgs.insert(0, {"role":"system","content":prompt_text})
    else:
        msgs[0]["content"] = prompt_text

def trim_history(max_turns:int):
    convo = get_active_convo()
    msgs = convo["messages"]
    if not msgs: return
    sys = msgs[0] if msgs and msgs[0]["role"]=="system" else None
    body = msgs[1:] if sys else msgs[:]
    limit = max_turns*2
    if len(body) > limit: body = body[-limit:]
    convo["messages"] = ([sys] if sys else []) + body

def extract_text_from_pdf(file_bytes:bytes)->str:
    if not HAS_PYPDF2: return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        return ""

def chunk_text(text:str, chunk_size=900, overlap=200):
    text = " ".join(text.split())
    if not text: return []
    out, start = [], 0
    while start < len(text):
        end = start + chunk_size
        out.append(text[start:end])
        if end >= len(text): break
        start = max(0, end-overlap)
    return out

def cosine_sim(A:np.ndarray,B:np.ndarray):
    A = A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-12)
    B = B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-12)
    return A @ B.T

def build_embeddings(client:OpenAI, chunks:list[str], model:str)->np.ndarray:
    if not chunks: return np.zeros((0,1536),dtype=np.float32)
    resp = client.embeddings.create(model=model, input=chunks)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def retrieve_context(query:str, top_k=4)->str:
    if not (st.session_state.rag_ready and st.session_state.rag_embeds is not None):
        return ""
    q = client.embeddings.create(model=st.session_state.rag_model, input=[query]).data[0].embedding
    qv = np.array([q], dtype=np.float32)
    sims = cosine_sim(qv, st.session_state.rag_embeds).flatten()
    idx = np.argsort(-sims)[:top_k]
    return "\n\n".join([st.session_state.rag_chunks[i] for i in idx])

def export_chat_as_txt(messages:list[dict])->bytes:
    return "\n".join(
        [f"[{m['role'].upper()}]\n{m['content']}\n" for m in messages if m["role"]!="system"]
    ).encode("utf-8")

def export_chat_as_json(messages:list[dict])->bytes:
    return json.dumps(
        [m for m in messages if m["role"] in ("user","assistant")],
        ensure_ascii=False, indent=2
    ).encode("utf-8")

def render_msg(role, content):
    who = "user" if role=="user" else "assistant"
    av  = "🙂" if role=="user" else "🐤"
    st.markdown(f'<div class="msg {who}"><div class="avatar">{av}</div>'
                f'<div class="bubble">{content}</div></div>', unsafe_allow_html=True)

def shorten_title(text: str, n:int=15)->str:
    t = " ".join(text.strip().split())
    return (t[:n] + "…") if len(t) > n else (t if t else "새 대화")

# ================= 상단: 타이틀 + 내보내기 =================
col_title, col_badges, col_menu = st.columns([6,3,1])
with col_title:
    st.markdown("### 🐤 **아이디어 챗봇**")
with col_menu:
    pop = st.popover("📥", use_container_width=True)
    with pop:
        st.markdown("**대화 내보내기**")
        active = get_active_convo()
        st.download_button("TXT로 저장", data=export_chat_as_txt(active["messages"]),
                           file_name="chat.txt", mime="text/plain", use_container_width=True)
        st.download_button("JSON으로 저장", data=export_chat_as_json(active["messages"]),
                           file_name="chat.json", mime="application/json", use_container_width=True)

# ================= 메인 상단: API Key 카드 =================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    default_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    openai_api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        value=default_key or "",
        help="환경변수/Secrets로 설정하지 않았다면 여기에 입력하세요. 세션에서만 사용됩니다."
    )
    st.markdown('</div>', unsafe_allow_html=True)

if not openai_api_key:
    st.info("사이드바 설정을 조정하기 전에, 상단의 **OpenAI API Key**를 입력해 주세요. (미입력 시 미리보기 모드)", icon="🗝️")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None
no_key = not bool(openai_api_key)

# ================= 사이드바: 새 대화 + 히스토리(리스트) + 설정 =================
with st.sidebar:
    st.markdown("## 💬 대화")

    # 새 대화
    if st.button("➕ 새 대화", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.active_id = new_id
        st.session_state.conversations.append({"id":new_id,"title":"새 대화","messages":[]})
        st.rerun()

    st.markdown("---")
    st.markdown("### 🕓 히스토리")

    # 라디오로 리스트 UI
    if st.session_state.conversations:
        ids = [c["id"] for c in st.session_state.conversations]
        titles = {c["id"]: ("💭 " + c["title"]) for c in st.session_state.conversations}
        # active index
        try:
            active_index = ids.index(st.session_state.active_id)
        except ValueError:
            active_index = len(ids)-1
        with st.container():
            st.markdown('<div class="sidebar-list">', unsafe_allow_html=True)
            selected_id = st.radio(
                label="대화 선택",
                options=ids,
                index=active_index,
                format_func=lambda _id: titles.get(_id, "💭 (제목 없음)"),
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        if selected_id != st.session_state.active_id:
            st.session_state.active_id = selected_id
            st.rerun()

    st.markdown("---")
    st.header("⚙️ 설정")
    model = st.selectbox("모델", ["gpt-4o", "gpt-4o-mini"], index=0)
    temperature = st.slider("창의성(temperature)", 0.0, 1.2, 0.7, 0.1)
    max_output_tokens = st.slider("응답 길이(max tokens)", 64, 4096, 1024, 64)
    stream_enable = st.toggle("실시간 스트리밍", value=True)
    st.markdown("---")
    preset = st.selectbox("말투/역할 프리셋", ["기본","친절한 튜터","초간단 요약봇","문장 다듬기(교정)"])
    preset_map = {
        "기본":"You are a helpful, concise assistant.",
        "친절한 튜터":"You are a friendly tutor. Explain step-by-step with small, clear paragraphs and examples.",
        "초간단 요약봇":"You summarize any input into 3 bullet points with the most essential facts only.",
        "문장 다듬기(교정)":"Rewrite the user's text with improved clarity, grammar, and natural tone while preserving meaning."
    }
    system_prompt = st.text_area("System prompt(세부 조정 가능)", value=preset_map[preset], height=100)
    st.markdown("---")
    max_turns_keep = st.slider("히스토리 보존 턴(질문/답변 쌍)", 5, 60, 30, 1)
    reset = st.button("🔄 현재 대화 리셋", use_container_width=True)

# 현재 대화 객체
active = get_active_convo()
if reset:
    active["messages"].clear()
    st.rerun()

# 상단 배지
with col_badges:
    st.markdown(
        f'<span class="badge">Model: <b>{model}</b></span> '
        f'<span class="badge">Temp: <b>{float(temperature):.2f}</b></span>',
        unsafe_allow_html=True
    )

# ================= 설명(접기) =================
with st.expander("설명 보기", expanded=False):
    st.markdown(
        "- OpenAI 모델을 사용합니다. API 키는 세션에서만 쓰이고 서버에 저장하지 않습니다.\n"
        "- 배포 시 **환경변수** 또는 **Streamlit Secrets** 사용을 권장합니다.\n"
        "- 업로드 파일(PDF/TXT)은 세션 메모리에만 저장됩니다."
    )

# ================= 📎 파일 업로드(선택) — 토글 =================
st.session_state.show_upload = st.toggle("📎 파일 업로드 (선택) — PDF/TXT 지원, 질의 응답에 활용 열기",
                                         value=st.session_state.show_upload,
                                         help="끄면 업로드/RAG UI가 숨겨집니다.")
if st.session_state.show_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "",  # 라벨 숨김
        type=["pdf","txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="PDF나 TXT를 올리면, 내용 기반으로 답변 품질이 향상돼요. 여러 파일 가능."
    )

    left, right = st.columns([3,2])
    with left:
        st.session_state.use_rag = st.toggle(
            "RAG 사용", value=st.session_state.get("use_rag", False),
            help="켜면 업로드 문서를 컨텍스트로 활용합니다."
        )
    with right:
        _pad, btn = st.columns([1,1])
        with btn:
            rebuild = st.button("📚 인덱스 생성/재생성", use_container_width=True)

    if rebuild and uploaded_files:
        if no_key:
            st.error("임베딩 생성에는 API 키가 필요합니다. 🔑")
        else:
            all_text=[]
            for f in uploaded_files:
                if f.type=="text/plain" or f.name.lower().endswith(".txt"):
                    text = f.read().decode("utf-8", errors="ignore")
                elif f.type=="application/pdf" or f.name.lower().endswith(".pdf"):
                    if not HAS_PYPDF2:
                        st.warning(f"'{f.name}': PyPDF2 미설치로 PDF 추출 불가(TXT만 지원)."); text=""
                    else:
                        text = extract_text_from_pdf(f.read())
                else:
                    text = ""
                if text: all_text.append(text)

            full_text = "\n\n".join(all_text)
            chunks = chunk_text(full_text, chunk_size=900, overlap=200)

            if not chunks:
                st.warning("추출 가능한 텍스트가 없습니다. 스캔 PDF는 텍스트가 없을 수 있어요.")
            else:
                with st.spinner("임베딩 생성 중…"):
                    vecs = build_embeddings(client, chunks, st.session_state.rag_model)
                st.session_state.rag_chunks = chunks
                st.session_state.rag_embeds = vecs
                st.session_state.rag_ready = True
                st.success(f"인덱스 생성 완료! 청크 {len(chunks)}개")

    st.markdown('</div>', unsafe_allow_html=True)
else:
    # 닫힐 때 선택 초기화(원하면 유지해도 됨)
    st.session_state.rag_ready = False
    st.session_state.rag_chunks = []
    st.session_state.rag_embeds = None

# ================= 대화 출력 =================
for m in active["messages"]:
    if m["role"] in ("user","assistant"):
        render_msg(m["role"], m["content"])

if not active["messages"]:
    st.markdown('<div class="card">❓ 먼저 질문을 입력해 보세요. 예) "이 PDF의 핵심 요약 3줄"</div>', unsafe_allow_html=True)

# ================= 입력 & 응답 =================
user_input = st.chat_input("무엇을 도와드릴까요? (Shift+Enter 줄바꿈)")
if user_input:
    ensure_system_message(system_prompt)
    trim_history(max_turns_keep)

    active["messages"].append({"role":"user","content":user_input})
    if active["title"] == "새 대화":
        active["title"] = shorten_title(user_input)
    render_msg("user", user_input)

    # RAG 컨텍스트
    additional_context = ""
    if st.session_state.get("use_rag", False):
        ctx = retrieve_context(user_input, top_k=4)
        if ctx:
            additional_context = (
                "You may use the following context extracted from the user's documents. "
                "If it is relevant, ground your answer in it. If not, ignore it.\n\n"
                f"=== BEGIN CONTEXT ===\n{ctx}\n=== END CONTEXT ==="
            )

    try:
        call_messages = list(active["messages"])
        if additional_context:
            call_messages.append({"role":"user","content":additional_context})

        if no_key:
            reply = "🔒 미리보기 모드: API Key를 입력하면 실제 답변이 생성됩니다."
            active["messages"].append({"role":"assistant","content":reply})
            render_msg("assistant", reply)
        else:
            if stream_enable:
                with st.spinner("생성 중…"):
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[{"role":m["role"],"content":m["content"]} for m in call_messages],
                        temperature=temperature, max_tokens=max_output_tokens, stream=True
                    )
                    response_text = st.write_stream(stream)
                active["messages"].append({"role":"assistant","content":response_text})
                render_msg("assistant", response_text)
            else:
                with st.spinner("생성 중…"):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role":m["role"],"content":m["content"]} for m in call_messages],
                        temperature=temperature, max_tokens=max_output_tokens, stream=False
                    )
                response_text = resp.choices[0].message.content
                active["messages"].append({"role":"assistant","content":response_text})
                render_msg("assistant", response_text)
    except (AuthenticationError, RateLimitError, APIError) as e:
        st.error(f"OpenAI API 오류: {e}")
    except Exception as e:
        st.exception(e)
