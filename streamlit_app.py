# app.py
import os, io, json
import numpy as np
import streamlit as st
from openai import OpenAI

# ============ 옵션: 예외 클래스(환경마다 달라서 안전 가드) ============
try:
    from openai import APIError, RateLimitError, AuthenticationError
except Exception:
    APIError = RateLimitError = AuthenticationError = Exception

# ============ PDF 추출 ============
try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# ================== 페이지/테마 ==================
st.set_page_config(page_title="아이디어 보드 Chatbot", page_icon="💡", layout="wide")

st.markdown("""
<style>
:root{ --brand:#F3CB2E; --ink:#222; --ink-weak:#666; --line:rgba(0,0,0,.08);
       --card:#fff; --bg:#F5F6F8; --muted:#EEF0F4; --radius:22px; }
html, body, [data-testid="stAppViewContainer"]{
  background:var(--bg);
  font-family: Pretendard, Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans KR", "Malgun Gothic", sans-serif;
  color:var(--ink);
}
.block-container{padding-top:1rem;padding-bottom:2rem;}
/* 사이드바 */
[data-testid="stSidebar"]{ background:var(--brand)!important; color:#1d1d1d;
  border-right:1px solid rgba(0,0,0,.15);}
[data-testid="stSidebar"] *{ color:#1d1d1d !important; }
[data-testid="stSidebar"] .stButton>button{
  border-radius:999px;border:1px solid rgba(0,0,0,.25);background:rgba(255,255,255,.3);
}
/* 카드/배지/툴바 */
.ui-card{ background:var(--card); border:1px solid var(--line); border-radius:26px;
  box-shadow:0 8px 26px rgba(0,0,0,.06); padding:18px 20px; }
.header-wrap{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px;}
.badge{ display:inline-flex; align-items:center; gap:6px; background:var(--muted);
  border:1px solid var(--line); border-radius:999px; padding:5px 10px; font-size:12px; color:#444;}
/* 입력 카드 */
.input-card{ padding:18px 20px 12px 20px; }
textarea{ font-size:18px !important; line-height:1.5 !important; }
.small-btn{ border:1px solid var(--line); background:#fff; border-radius:12px; padding:8px 10px; display:inline-block;}
.send-btn{ background:#231F20; color:#fff; border:none; border-radius:999px; padding:12px 18px; font-weight:700; }
.hr{ border:none; height:1px; background:var(--line); margin:16px 0 10px; }
/* 말풍선 */
.msg{ display:flex; gap:10px; margin:10px 0;}
.bubble{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:12px 14px;}
.user .bubble{ background:#F8FAFF; border-color:rgba(91,141,239,.35); }
.assistant .bubble{ background:#fff; }
.avatar{ width:30px;height:30px;border-radius:50%; background:var(--muted);
  display:flex;align-items:center;justify-content:center;}
/* ChatInput 살짝 고정 느낌 */
div[data-testid="stChatInput"]{ position:sticky; bottom:0; z-index:10; background:rgba(255,255,255,.85);
  backdrop-filter:blur(6px); border-top:1px solid var(--line);}
</style>
""", unsafe_allow_html=True)

# ================== 세션 기본값 ==================
if "messages" not in st.session_state:
    st.session_state.messages = []            # [{"role":"system|user|assistant","content":...}]
    st.session_state.has_system = False
st.session_state.setdefault("rag_ready", False)
st.session_state.setdefault("rag_chunks", [])
st.session_state.setdefault("rag_embeds", None)  # np.array (N,D)
st.session_state.setdefault("rag_model", "text-embedding-3-small")
st.session_state.setdefault("use_rag", False)
st.session_state.setdefault("show_upload", False)

# ================== 도우미 ==================
def ensure_system_message(prompt_text: str):
    if not st.session_state.has_system:
        st.session_state.messages.insert(0, {"role":"system","content":prompt_text})
        st.session_state.has_system = True
    else:
        st.session_state.messages[0]["content"] = prompt_text

def trim_history(max_turns:int):
    msgs = st.session_state.messages
    if not msgs: return
    sys = msgs[0] if msgs and msgs[0]["role"]=="system" else None
    body = msgs[1:] if sys else msgs[:]
    limit = max_turns*2
    if len(body)>limit: body = body[-limit:]
    st.session_state.messages = ([sys] if sys else []) + body

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not HAS_PYPDF2: return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception: return ""

def chunk_text(text:str, chunk_size=900, overlap=200):
    text = " ".join(text.split())
    if not text: return []
    out, start = [], 0
    while start < len(text):
        end = start + chunk_size
        out.append(text[start:end])
        if end >= len(text): break
        start = max(0, end - overlap)
    return out

def cosine_sim(A:np.ndarray,B:np.ndarray):
    A = A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-12)
    B = B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-12)
    return A @ B.T

def build_embeddings(client:OpenAI, chunks:list[str], model:str)->np.ndarray:
    if not chunks: return np.zeros((0,1536),dtype=np.float32)
    resp = client.embeddings.create(model=model, input=chunks)
    return np.array([d.embedding for d in resp.data],dtype=np.float32)

def retrieve_context(query:str, top_k=4) -> str:
    if not (st.session_state.rag_ready and st.session_state.rag_embeds is not None):
        return ""
    try:
        q = client.embeddings.create(model=st.session_state.rag_model, input=[query]).data[0].embedding
        qv = np.array([q],dtype=np.float32)
        sims = cosine_sim(qv, st.session_state.rag_embeds).flatten()
        idx = np.argsort(-sims)[:top_k]
        return "\n\n".join([st.session_state.rag_chunks[i] for i in idx])
    except Exception:
        return ""

def export_chat_as_txt(messages:list[dict])->bytes:
    lines=[]
    for m in messages:
        if m.get("role")=="system": continue
        lines.append(f"[{m['role'].upper()}]\n{m.get('content','')}\n")
    return "\n".join(lines).encode("utf-8")

def export_chat_as_json(messages:list[dict])->bytes:
    payload=[m for m in messages if m.get("role") in ("user","assistant")]
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

def render_msg(role, content):
    who = "user" if role=="user" else "assistant"
    av  = "🙂" if role=="user" else "🤖"
    st.markdown(
        f'<div class="msg {who}"><div class="avatar">{av}</div>'
        f'<div class="bubble">{content}</div></div>', unsafe_allow_html=True
    )

# ================== 타이틀 & 안내 ==================
st.markdown("## 🧠 아이디어 보드 Chatbot")
default_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not default_key:
    st.info("🔑 **사이드바에 OpenAI API Key를 입력하세요.** (없어도 미리보기로 UI는 사용 가능)", icon="🗝️")

# ================== 사이드바 ==================
with st.sidebar:
    st.markdown("## 설정")
    # API Key 입력
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=default_key or "")
    # 모델/파라미터
    model = st.selectbox("모델", ["gpt-4o","gpt-4o-mini"], index=0)
    temperature = st.slider("창의성(temperature)", 0.0, 1.2, 0.7, 0.1)
    max_output_tokens = st.slider("응답 길이(max tokens)", 64, 4096, 1024, 64)
    stream_enable = st.toggle("실시간 스트리밍", value=True)
    st.markdown("---")
    # 프리셋
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
    reset = st.button("🔄 새 대화 시작", use_container_width=True)

if reset:
    st.session_state.clear()
    st.rerun()

# 키/클라이언트
client = OpenAI(api_key=openai_api_key) if openai_api_key else None
no_key = not bool(openai_api_key)

# ================== 헤더 상태/내보내기 ==================
st.markdown('<div class="ui-card header-wrap">', unsafe_allow_html=True)
st.markdown(
    f'<div><span class="badge">Model: {model}</span> '
    f'<span class="badge">Temp: {temperature:.2f}</span></div>',
    unsafe_allow_html=True
)
c1, c2 = st.columns([1,1])
with c1:
    st.download_button("TXT 내보내기", data=export_chat_as_txt(st.session_state.messages),
                       file_name="chat.txt", mime="text/plain", use_container_width=True)
with c2:
    st.download_button("JSON 내보내기", data=export_chat_as_json(st.session_state.messages),
                       file_name="chat.json", mime="application/json", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================== 파일 업로드(토글) + RAG ==================
st.session_state.show_upload = st.toggle("📎 파일 업로드 섹션 열기", value=st.session_state.show_upload)
if st.session_state.show_upload:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown("#### 📎 파일 업로드 (선택) — PDF/TXT 지원, 질의·응답에 활용")

    uploaded_files = st.file_uploader(
        "여기에 PDF나 TXT를 올리면, 내용 기반으로 답변 품질이 향상돼요. 여러 파일 가능.",
        type=["pdf","txt"], accept_multiple_files=True
    )
    left, right = st.columns([3,2])
    with left:
        st.session_state.use_rag = st.toggle("RAG 사용", value=st.session_state.get("use_rag", False),
                                             help="켜면 업로드 문서를 컨텍스트로 활용합니다.")
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
                    text = extract_text_from_pdf(f.read()) if HAS_PYPDF2 else ""
                    if not text and not HAS_PYPDF2:
                        st.warning(f"'{f.name}': PyPDF2 미설치로 PDF 추출 불가(TXT만 지원).")
                else:
                    text=""
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
    # 섹션 닫을 때 선택적으로 리셋
    st.session_state.rag_ready = False
    st.session_state.rag_chunks = []
    st.session_state.rag_embeds = None

# ================== 히스토리 렌더링 ==================
for m in st.session_state.messages:
    if m["role"] in ("user","assistant"):
        render_msg(m["role"], m["content"])

if not st.session_state.messages:
    st.markdown('<div class="ui-card">❓ 먼저 질문을 입력해 보세요. 예) "이 PDF의 핵심 요약 3줄"</div>', unsafe_allow_html=True)

# ================== 입력 카드(폼) ==================
with st.container():
    st.markdown('<div class="ui-card input-card">', unsafe_allow_html=True)
    with st.form("prompt_form", clear_on_submit=False):
        user_input = st.text_area(
            "아이디어를 설명하거나 주사위를 굴려 아이디어를 얻으세요.",
            value="", height=160, label_visibility="collapsed",
            placeholder="아이디어를 설명하거나 주사위를 굴려 아이디어를 얻으세요."
        )
        cA, cB, cC, cSend = st.columns([1,1,6,1])
        with cA:  st.markdown('<div class="small-btn">🖼️ 이미지 추가 (demo)</div>', unsafe_allow_html=True)
        with cB:  st.markdown('<div class="small-btn">🎲 랜덤 (demo)</div>', unsafe_allow_html=True)
        with cC:  pass
        with cSend:
            submitted = st.form_submit_button("➜", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ================== 생성/응답 ==================
if submitted and user_input.strip():
    ensure_system_message(system_prompt)
    trim_history(max_turns_keep)

    st.session_state.messages.append({"role":"user","content":user_input})
    render_msg("user", user_input)

    # RAG 컨텍스트
    additional_context = ""
    if st.session_state.get("use_rag", False):
        ctx = retrieve_context(user_input, top_k=4)
        if ctx:
            additional_context = (
                "You may use the following context extracted from the user's documents. "
                "If it is relevant, ground your answer in it.\n\n"
                f"=== BEGIN CONTEXT ===\n{ctx}\n=== END CONTEXT ==="
            )

    try:
        call_messages = list(st.session_state.messages)
        if additional_context:
            call_messages.append({"role":"user","content":additional_context})

        if no_key:
            reply = "🔒 미리보기 모드: 사이드바에 API Key를 입력하면 실제 답변을 생성합니다."
            st.session_state.messages.append({"role":"assistant","content":reply})
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
                st.session_state.messages.append({"role":"assistant","content":response_text})
                render_msg("assistant", response_text)
            else:
                with st.spinner("생성 중…"):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role":m["role"],"content":m["content"]} for m in call_messages],
                        temperature=temperature, max_tokens=max_output_tokens, stream=False
                    )
                response_text = resp.choices[0].message.content
                st.session_state.messages.append({"role":"assistant","content":response_text})
                render_msg("assistant", response_text)
                if getattr(resp,"usage",None):
                    st.markdown(
                        f'<div class="badge">🧮 tokens: {resp.usage.total_tokens} '
                        f'(in {resp.usage.prompt_tokens} / out {resp.usage.completion_tokens})</div>',
                        unsafe_allow_html=True
                    )
    except (AuthenticationError, RateLimitError, APIError) as e:
        st.error(f"OpenAI API 오류: {e}")
    except Exception as e:
        st.exception(e)
