# app.py
import os
import io
import json
import numpy as np
import streamlit as st
from openai import OpenAI

# 예외 클래스 (환경에 따라 임포트가 다를 수 있어 안전 가드)
try:
    from openai import APIError, RateLimitError, AuthenticationError
except Exception:
    APIError = RateLimitError = AuthenticationError = Exception

# PDF 텍스트 추출: PyPDF2가 없으면 TXT만 지원
try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# ----------------------------
# 기본 세팅
# ----------------------------
st.set_page_config(page_title="💬 나의 첫번째 Chatbot", page_icon="💬", layout="wide")
st.title("💬 나의 첫번째 Chatbot")

# 세션의 API 키 기본값 생성 (환경변수/Secrets 우선)
default_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = default_key  # 없으면 None

# 상단 안내 (키 없을 때만)
if not st.session_state.openai_api_key:
    st.info("🔑 **사이드바에 OpenAI API Key를 입력하세요.**", icon="🗝️")

# ----------------------------
# 사이드바: 설정 (이 블록만 사용)
# ----------------------------
with st.sidebar:
    st.header("⚙️ 설정")

    # API 키 입력 (세션 기반)
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key or "",
        help="환경변수/Secrets가 없으면 여기에 입력"
    )

    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="일반 대화: gpt-4o / 비용절감: gpt-4o-mini"
    )
    temperature = st.slider("Temperature(창의성)", 0.0, 1.2, 0.7, 0.1)
    max_output_tokens = st.slider("Max output tokens(응답 길이)", 64, 4096, 1024, 64)
    stream_enable = st.toggle("실시간 스트리밍", value=True, help="끄면 응답 후 토큰 사용량을 표시합니다.")
    st.divider()

    st.subheader("Assistant 스타일 프리셋")
    preset = st.selectbox(
        "말투/역할 프리셋",
        ["기본", "친절한 튜터", "초간단 요약봇", "문장 다듬기(교정)"],
        index=0
    )
    preset_map = {
        "기본": "You are a helpful, concise assistant.",
        "친절한 튜터": "You are a friendly tutor. Explain step-by-step with small, clear paragraphs and examples.",
        "초간단 요약봇": "You summarize any input into 3 bullet points with the most essential facts only.",
        "문장 다듬기(교정)": "Rewrite the user's text with improved clarity, grammar, and natural tone while preserving meaning."
    }
    system_prompt = st.text_area(
        "System prompt(세부 조정 가능)",
        value=preset_map.get(preset, preset_map["기본"]),
        height=100
    )

    st.subheader("대화 관리")
    max_turns_keep = st.slider("히스토리 보존 턴(질문/답변 쌍)", 5, 60, 30, 1)
    reset = st.button("🔄 새 대화 시작")
    st.caption("너무 길어지면 비용↑/속도↓ → 오래된 기록은 자동 트림")

# reset 즉시 적용
if reset:
    st.session_state.clear()
    st.rerun()

# 키/클라이언트 확정
openai_api_key = st.session_state.openai_api_key
no_key = not openai_api_key
client = OpenAI(api_key=openai_api_key) if not no_key else None

# 설명 박스 (접기)
with st.expander("설명 보기", expanded=False):
    st.markdown(
        "- OpenAI 모델을 사용합니다. API 키는 세션에서만 쓰이고 서버에 저장하지 않습니다.\n"
        "- 배포 시 **환경변수** 또는 **Streamlit Secrets** 사용을 권장합니다.\n"
        "- 업로드 파일(PDF/TXT)은 세션 메모리에만 저장됩니다."
    )

# ----------------------------
# 전역 UI 테마 (폰트/카드/배지/말풍선/인풋 고정)
# ----------------------------
st.markdown("""
<style>
:root {
  --brand-text:#222; --brand-accent:#5B8DEF;
  --bg-card:#fff; --bg-soft:#F6F8FB; --line-soft:rgba(0,0,0,.07); --radius-xl:16px;
}
html, body, [data-testid="stAppViewContainer"] {
  font-family: Pretendard, Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
  color: var(--brand-text);
}
h1,h2,h3 { letter-spacing:-.01em }
.block-container { padding-top: 1rem; padding-bottom:2.5rem; }
.ui-card{ background:var(--bg-card); border:1px solid var(--line-soft); border-radius:var(--radius-xl); box-shadow:0 6px 18px rgba(0,0,0,.04); padding:16px 18px; }
.ui-toolbar{ display:flex; align-items:center; gap:12px; }
.ui-toolbar .right{ margin-left:auto; display:flex; gap:8px; align-items:center; }
.badge{ display:inline-flex; align-items:center; gap:6px; background:var(--bg-soft); border:1px solid var(--line-soft); border-radius:999px; padding:4px 10px; font-size:12px;}
.chat-bubble{ border:1px solid var(--line-soft); border-radius:18px; padding:12px 14px; margin:6px 0; background:#fff;}
.chat-bubble.user{ background:#F1F6FF; border-color:rgba(45,116,255,.2); }
.chat-bubble.assist{ background:#F9FAFB; }
.chat-row{ display:flex; gap:10px; align-items:flex-start;}
.chat-avatar{ width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:#EEF0F4; font-size:14px;}
div[data-testid="stChatInput"]{ position:sticky; bottom:0; z-index:10; background:rgba(255,255,255,.85); backdrop-filter:blur(6px); border-top:1px solid var(--line-soft);}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 세션 상태 초기화
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"system"/"user"/"assistant","content":...}]
    st.session_state.has_system = False
st.session_state.setdefault("rag_ready", False)
st.session_state.setdefault("rag_chunks", [])
st.session_state.setdefault("rag_embeds", None)    # np.array (N,D)
st.session_state.setdefault("rag_model", "text-embedding-3-small")
st.session_state.setdefault("use_rag", False)

# ----------------------------
# 도우미 함수
# ----------------------------
def ensure_system_message(prompt_text: str):
    if not st.session_state.has_system:
        st.session_state.messages.insert(0, {"role": "system", "content": prompt_text})
        st.session_state.has_system = True
    else:
        st.session_state.messages[0]["content"] = prompt_text

def trim_history(max_turns: int):
    msgs = st.session_state.messages
    if not msgs: return
    sys = msgs[0] if msgs and msgs[0]["role"] == "system" else None
    body = msgs[1:] if sys else msgs[:]
    limit = max_turns * 2
    if len(body) > limit: body = body[-limit:]
    st.session_state.messages = ([sys] if sys else []) + body

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not HAS_PYPDF2: return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        return ""

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200):
    text = " ".join(text.split())
    if not text: return []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text): break
        start = max(0, end - overlap)
    return chunks

def cosine_sim(A: np.ndarray, B: np.ndarray):
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

def build_embeddings(client: OpenAI, chunks: list[str], embed_model: str) -> np.ndarray:
    if not chunks: return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=embed_model, input=chunks)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)

def retrieve_context(query: str, top_k: int = 4) -> str:
    if not (st.session_state.rag_ready and st.session_state.rag_embeds is not None):
        return ""
    try:
        q_emb = client.embeddings.create(model=st.session_state.rag_model, input=[query]).data[0].embedding
        q_vec = np.array([q_emb], dtype=np.float32)
        sims = cosine_sim(q_vec, st.session_state.rag_embeds).flatten()
        idx = np.argsort(-sims)[:top_k]
        return "\n\n".join([st.session_state.rag_chunks[i] for i in idx])
    except Exception:
        return ""

def export_chat_as_txt(messages: list[dict]) -> bytes:
    lines = []
    for m in messages:
        r = m.get("role", "")
        if r == "system": continue
        lines.append(f"[{r.upper()}]\n{m.get('content','')}\n")
    return "\n".join(lines).encode("utf-8")

def export_chat_as_json(messages: list[dict]) -> bytes:
    payload = [m for m in messages if m.get("role") in ("user", "assistant")]
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

# 미리보기 모드 알림
if no_key:
    st.warning("API 키가 없어 **미리보기 모드**로 동작합니다. 응답/임베딩은 생성되지 않습니다. 🔒")

# ----------------------------
# 헤더 바 (상태 배지 + 내보내기 버튼)
# ----------------------------
st.markdown('<div class="ui-card ui-toolbar">', unsafe_allow_html=True)
st.markdown("### 💬 대화 세션")
st.markdown(
    f'<span class="badge">Model: <b>{model}</b></span> '
    f'<span class="badge">Temp: <b>{float(temperature):.2f}</b></span>',
    unsafe_allow_html=True
)
st.markdown('<span class="right"></span>', unsafe_allow_html=True)

c1, c2 = st.columns([1,1])
with c1:
    st.download_button("TXT 내보내기", data=export_chat_as_txt(st.session_state.messages),
                       file_name="chat.txt", mime="text/plain", use_container_width=True)
with c2:
    st.download_button("JSON 내보내기", data=export_chat_as_json(st.session_state.messages),
                       file_name="chat.json", mime="application/json", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 📎 파일 업로드 섹션 (토글로 열고/닫기) + 카드 UI
# ----------------------------
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

st.session_state.show_upload = st.toggle("📎 파일 업로드 섹션 열기", value=st.session_state.show_upload)

if st.session_state.show_upload:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown("#### 📎 파일 업로드 (선택) — PDF/TXT 지원, 질의·응답에 활용")

    uploaded_files = st.file_uploader(
        "여기에 PDF나 TXT를 올리면, 내용 기반으로 답변 품질이 향상돼요. 여러 파일 가능.",
        type=["pdf", "txt"], accept_multiple_files=True
    )

    left, right = st.columns([3, 2])
    with left:
        use_rag = st.toggle("RAG 사용", value=st.session_state.get("use_rag", False),
                            help="켜면 업로드 문서를 컨텍스트로 활용합니다.")
        st.session_state.use_rag = use_rag
    with right:
        _pad, btn = st.columns([1,1])
        with btn:
            rebuild = st.button("📚 인덱스 생성/재생성", use_container_width=True)

    if rebuild and uploaded_files:
        if no_key:
            st.error("임베딩 생성에는 API 키가 필요합니다. 🔑")
        else:
            all_text = []
            for f in uploaded_files:
                if f.type == "text/plain" or f.name.lower().endswith(".txt"):
                    text = f.read().decode("utf-8", errors="ignore")
                elif f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                    if not HAS_PYPDF2:
                        st.warning(f"'{f.name}' → PyPDF2 미설치로 PDF 추출 불가(TXT만 지원)."); text = ""
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
    # 섹션 닫힐 때 선택적으로 RAG 상태 초기화
    st.session_state.rag_ready = False
    st.session_state.rag_chunks = []
    st.session_state.rag_embeds = None

# ----------------------------
# 말풍선 렌더러 & 기존 히스토리 출력
# ----------------------------
def render_msg(role:str, content:str):
    kind = "user" if role=="user" else "assist"
    avatar = "🧑‍💻" if role=="user" else "🤖"
    st.markdown(
        f'<div class="chat-row"><div class="chat-avatar">{avatar}</div>'
        f'<div class="chat-bubble {kind}">{content}</div></div>', unsafe_allow_html=True
    )

for m in st.session_state.messages:
    if m["role"] in ("user", "assistant"):
        render_msg(m["role"], m["content"])

# 빈 상태 안내
if not st.session_state.messages:
    st.markdown('<div class="ui-card">❓ 먼저 질문을 입력해 보세요. 예) "이 PDF의 핵심 요약 3줄"</div>', unsafe_allow_html=True)

# ----------------------------
# 입력 & 응답
# ----------------------------
user_input = st.chat_input("무엇을 도와드릴까요? (Shift+Enter 줄바꿈)")
if user_input:
    ensure_system_message(system_prompt)
    trim_history(max_turns_keep)

    st.session_state.messages.append({"role":"user","content":user_input})
    render_msg("user", user_input)

    # 문서 컨텍스트
    additional_context = ""
    if st.session_state.get("use_rag", False):
        ctx = retrieve_context(user_input, top_k=4)
        if ctx:
            additional_context = (
                "Use the following context from user's documents when relevant.\n\n"
                f"=== BEGIN CONTEXT ===\n{ctx}\n=== END CONTEXT ==="
            )

    try:
        call_messages = list(st.session_state.messages)
        if additional_context:
            call_messages.append({"role":"user","content":additional_context})

        if no_key:
            render_msg("assistant", "🔒 미리보기 모드: API 키를 입력하면 실제 답변이 생성됩니다.")
        else:
            if stream_enable:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": m["role"], "content": m["content"]} for m in call_messages],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    stream=True,
                )
                with st.spinner("생성 중…"):
                    response_text = st.write_stream(stream)
                st.session_state.messages.append({"role":"assistant","content":response_text})
                render_msg("assistant", response_text)
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": m["role"], "content": m["content"]} for m in call_messages],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    stream=False,
                )
                response_text = resp.choices[0].message.content
                st.session_state.messages.append({"role":"assistant","content":response_text})
                render_msg("assistant", response_text)
                if getattr(resp, "usage", None):
                    in_tok = resp.usage.prompt_tokens
                    out_tok = resp.usage.completion_tokens
                    tot_tok = resp.usage.total_tokens
                    st.markdown(f'<span class="badge">🧮 tokens: {tot_tok} (in {in_tok} / out {out_tok})</span>', unsafe_allow_html=True)

    except (AuthenticationError, RateLimitError, APIError) as e:
        st.error(f"OpenAI API 오류: {e}")
    except Exception as e:
        st.exception(e)
