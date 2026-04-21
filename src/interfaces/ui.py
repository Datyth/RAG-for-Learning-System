"""Streamlit UI — pure frontend, calls FastAPI backend via HTTP.

Run with:
    uv run streamlit run src/interfaces/ui.py

Requires the API to be running:
    uv run rag-api

Override API URL with RAG_API_URL env var (default: http://localhost:8000).
All user-facing text is in Vietnamese.
"""

import html
import os

import httpx
import streamlit as st

from src.export import flashcards_to_markdown, quiz_to_markdown, summary_to_markdown
from src.interfaces.styles import GLOBAL_CSS
from src.schemas import Citation, FlashcardSet, QuizSet, RagAnswer, RetrievedChunk, Summary

_API = os.environ.get("RAG_API_URL", "http://localhost:8000")
_ALL_DOCS = "(Tất cả tài liệu)"
_ALL_PAGES = "(Tất cả trang)"


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    """Synchronous API call; stops page render on connection failure."""
    try:
        return httpx.request(method, f"{_API}{path}", timeout=120, **kwargs)
    except httpx.ConnectError:
        st.error(f"Cannot connect to API at {_API}. Run `uv run rag-api` first.")
        st.stop()


def _filters_json(document: str | None, page: int | None) -> dict | None:
    f: dict = {}
    if document:
        f["filename"] = document
    if page is not None:
        f["page"] = page
    return f or None


def _init_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("uploaded_files", set())


def _render_citations(citations: list[Citation]) -> None:
    if not citations:
        return
    with st.expander("Nguồn trích dẫn", expanded=False):
        for c in citations:
            section = f" — {c.section}" if c.section else ""
            st.markdown(f"- **[{c.source_marker}]** `{c.filename}` (trang {c.page}){section}")


def _render_chunks(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        return
    with st.expander("Chi tiết đoạn văn được trích", expanded=False):
        for i, c in enumerate(chunks, start=1):
            meta = c.metadata
            st.markdown(f"**[S{i}] {meta.filename} — trang {meta.page}** · điểm: {c.score:.3f}")
            st.code(c.text.strip(), language="text")


def _sidebar() -> tuple[str | None, int | None]:
    st.sidebar.title("📚 RAG Learning")
    st.sidebar.subheader("Tải tài liệu PDF")
    uploaded = st.sidebar.file_uploader(
        "Chọn một hoặc nhiều PDF", type=["pdf"], accept_multiple_files=True, key="uploader"
    )
    if uploaded:
        for f in uploaded:
            if f.name in st.session_state.uploaded_files:
                continue
            with st.sidebar.status(f"Đang nạp {f.name}...", expanded=False):
                r = _api("POST", "/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
                if r.status_code == 400:
                    st.sidebar.error(f"Lỗi: {r.json().get('detail', 'Unknown error')}")
                    continue
                info = r.json()
            st.session_state.uploaded_files.add(f.name)
            st.sidebar.success(f"Đã nạp {info['filename']} · {info['chunks_indexed']} đoạn")

    st.sidebar.subheader("Bộ lọc tài liệu")
    try:
        docs = _api("GET", "/documents").json()
    except Exception as exc:
        st.sidebar.error(f"Không đọc được tài liệu: {exc}")
        docs = []

    doc_map = {d["filename"]: d for d in docs}
    selected = st.sidebar.selectbox(
        "Tài liệu", [_ALL_DOCS] + list(doc_map.keys()), index=0, key="doc_select"
    )

    page: int | None = None
    if selected != _ALL_DOCS:
        pages = doc_map[selected]["pages"]
        chosen = st.sidebar.selectbox(
            "Trang", [_ALL_PAGES] + [str(p) for p in pages], index=0, key="page_select"
        )
        if chosen != _ALL_PAGES:
            page = int(chosen)
    else:
        st.sidebar.caption("Chọn một tài liệu để lọc theo trang.")

    if docs:
        with st.sidebar.expander(f"Đang có {len(docs)} tài liệu trong kho"):
            for d in docs:
                st.write(f"- `{d['filename']}` · {d['page_count']} trang · {d['chunk_count']} đoạn")

    return None if selected == _ALL_DOCS else selected, page


def _tab_chat(document: str | None, page: int | None) -> None:
    st.subheader("Hỏi đáp có trích dẫn")
    st.caption("Đặt câu hỏi về nội dung đã nạp. Bộ lọc ở thanh bên sẽ thu hẹp phạm vi tìm kiếm.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                _render_citations([Citation(**c) for c in msg["citations"]])
            if msg.get("chunks"):
                _render_chunks([RetrievedChunk(**c) for c in msg["chunks"]])

    q = st.chat_input("Đặt câu hỏi về tài liệu...")
    if not q:
        return

    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            r = _api("POST", "/ask", json={"question": q, "filters": _filters_json(document, page)})
            res = RagAnswer.model_validate(r.json())
        st.markdown(res.answer)
        _render_citations(res.citations)
        _render_chunks(res.chunks)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": res.answer,
         "citations": [c.model_dump() for c in res.citations],
         "chunks": [c.model_dump() for c in res.chunks]}
    )


def _tab_summary(document: str | None, page: int | None) -> None:
    st.subheader("Tóm tắt tài liệu")
    with st.form("summary_form"):
        query = st.text_input("Chủ đề hướng dẫn (tuỳ chọn)")
        k = st.number_input("Số đoạn truy xuất (k)", 1, 64, 12)
        submit = st.form_submit_button("Tạo tóm tắt", use_container_width=True, type="primary")
    if not submit:
        return

    with st.spinner("Đang tóm tắt..."):
        r = _api("POST", "/summarize", json={
            "document": document, "query": query or None,
            "filters": _filters_json(None, page), "k": int(k),
        })
    if r.status_code == 422:
        st.error(f"Không thể tóm tắt: {r.json().get('detail')}")
        return
    res = Summary.model_validate(r.json())

    if not res.summary:
        st.warning("Không tìm thấy nội dung phù hợp để tóm tắt.")
        return

    st.markdown(res.summary)
    if res.key_points:
        st.markdown("**Các ý chính:**")
        for kp in res.key_points:
            st.markdown(f"- {kp}")
    _render_citations(res.citations)
    st.divider()
    col_a, col_b = st.columns(2)
    col_a.download_button("Tải JSON", res.model_dump_json(indent=2), file_name="summary.json", mime="application/json")
    col_b.download_button("Tải Markdown", summary_to_markdown(res), file_name="summary.md", mime="text/markdown")
    st.session_state.history.append({"kind": "summary", "target": res.target})


def _clear_quiz_state() -> None:
    stale = [k for k in st.session_state if k.startswith(("quiz_q_", "quiz_ans_"))]
    for k in stale:
        del st.session_state[k]


def _tab_quiz(document: str | None, page: int | None) -> None:
    st.subheader("Tạo bộ câu hỏi")

    has_data = "quiz_res" in st.session_state
    with st.expander("Tạo quiz mới", expanded=not has_data):
        with st.form("quiz_form"):
            query = st.text_input("Chủ đề (tuỳ chọn)")
            c1, c2 = st.columns(2)
            count = c1.number_input("Số câu hỏi", 1, 30, 8)
            retrieval_k = c2.number_input("Số đoạn truy xuất", 1, 64, 16)
            if st.form_submit_button("Tạo quiz", use_container_width=True, type="primary"):
                with st.spinner("Đang tạo quiz..."):
                    r = _api("POST", "/quiz", json={
                        "document": document, "query": query or None,
                        "filters": _filters_json(None, page),
                        "count": int(count), "k": int(retrieval_k),
                    })
                if r.status_code == 422:
                    st.error(f"Không thể tạo quiz: {r.json().get('detail')}")
                else:
                    res = QuizSet.model_validate(r.json())
                    st.session_state.quiz_res = res
                    st.session_state.quiz_submitted = False
                    _clear_quiz_state()
                    st.session_state.history.append(
                        {"kind": "quiz", "target": res.target, "count": len(res.items)}
                    )

    if "quiz_res" not in st.session_state:
        return

    res = st.session_state.quiz_res
    submitted = st.session_state.get("quiz_submitted", False)

    if submitted:
        correct = sum(
            1 for i, item in enumerate(res.items)
            if st.session_state.get(f"quiz_ans_{i}") == item.correct_index
        )
        pct = int(100 * correct / len(res.items))
        col_score, col_btn = st.columns([3, 1])
        col_score.metric("Kết quả", f"{correct}/{len(res.items)} câu đúng", f"{pct}%")
        with col_btn:
            st.write("")
            if st.button("Làm lại", use_container_width=True):
                st.session_state.quiz_submitted = False
                _clear_quiz_state()
                st.rerun()

    for i, item in enumerate(res.items):
        meta_bits = [x for x in [item.topic, item.difficulty] if x]
        suffix = f" · _{' · '.join(meta_bits)}_" if meta_bits else ""
        letter_map = {opt: f"{chr(65 + j)}. {opt}" for j, opt in enumerate(item.options)}
        with st.container(border=True):
            st.markdown(f"**Câu {i + 1}.**{suffix}")
            st.markdown(item.question)
            st.radio(
                f"q{i}", item.options, key=f"quiz_q_{i}",
                format_func=lambda opt, m=letter_map: m[opt],
                label_visibility="collapsed", index=None, disabled=submitted,
            )
            if submitted:
                ans = st.session_state.get(f"quiz_ans_{i}")
                correct_label = f"**{chr(65 + item.correct_index)}. {item.options[item.correct_index]}**"
                if ans == item.correct_index:
                    st.success(f"Đúng! Đáp án: {correct_label}")
                elif ans is not None:
                    st.error(f"Sai. Đáp án đúng: {correct_label}")
                else:
                    st.warning(f"Chưa chọn. Đáp án: {correct_label}")
                if item.explanation:
                    with st.expander("Giải thích"):
                        st.markdown(item.explanation)
            elif item.source_markers:
                st.caption("Nguồn: " + ", ".join(item.source_markers))

    if not submitted:
        if st.button("Nộp bài", type="primary"):
            for i, item in enumerate(res.items):
                val = st.session_state.get(f"quiz_q_{i}")
                st.session_state[f"quiz_ans_{i}"] = item.options.index(val) if val is not None else None
            st.session_state.quiz_submitted = True
            st.rerun()

    _render_citations(res.citations)
    st.divider()
    col_a, col_b = st.columns(2)
    col_a.download_button("Tải JSON", res.model_dump_json(indent=2), file_name="quiz.json", mime="application/json")
    col_b.download_button("Tải Markdown", quiz_to_markdown(res), file_name="quiz.md", mime="text/markdown")


def _fc_card_html(face: str, side: str, topic: str | None, hint: str | None, flipped: bool) -> str:
    cls = "fc-back" if flipped else "fc-front"
    topic_html = f'<p class="fc-topic">{html.escape(topic)}</p>' if (topic and not flipped) else ""
    hint_html = f'<p class="fc-hint">{html.escape(hint)}</p>' if (hint and flipped) else ""
    return (
        f'<div class="fc-card {cls}">'
        f'<p class="fc-side">{side}</p>'
        f'{topic_html}'
        f'<p class="fc-text">{html.escape(face).replace(chr(10), "<br>")}</p>'
        f'{hint_html}'
        f'</div>'
    )


def _tab_flashcards(document: str | None, page: int | None) -> None:
    st.subheader("Tạo thẻ ghi nhớ")

    has_data = "fc_res" in st.session_state
    with st.expander("Tạo bộ thẻ mới", expanded=not has_data):
        with st.form("fc_form"):
            query = st.text_input("Chủ đề (tuỳ chọn)")
            c1, c2 = st.columns(2)
            count = c1.number_input("Số thẻ", 1, 40, 15)
            retrieval_k = c2.number_input("Số đoạn truy xuất", 1, 64, 16)
            if st.form_submit_button("Tạo thẻ", use_container_width=True, type="primary"):
                with st.spinner("Đang tạo flashcards..."):
                    r = _api("POST", "/flashcards", json={
                        "document": document, "query": query or None,
                        "filters": _filters_json(None, page),
                        "count": int(count), "k": int(retrieval_k),
                    })
                if r.status_code == 422:
                    st.error(f"Không thể tạo thẻ: {r.json().get('detail')}")
                else:
                    res = FlashcardSet.model_validate(r.json())
                    st.session_state.fc_res = res
                    st.session_state.fc_idx = 0
                    st.session_state.fc_flipped = False
                    st.session_state.history.append(
                        {"kind": "flashcards", "target": res.target, "count": len(res.cards)}
                    )

    if "fc_res" not in st.session_state:
        return

    res = st.session_state.fc_res
    cards = res.cards
    idx = st.session_state.get("fc_idx", 0)
    flipped = st.session_state.get("fc_flipped", False)
    card = cards[idx]

    st.progress((idx + 1) / len(cards), text=f"Thẻ {idx + 1} / {len(cards)}")
    st.markdown(
        _fc_card_html(
            card.back if flipped else card.front,
            "Mặt sau" if flipped else "Mặt trước",
            card.topic, card.hint, flipped,
        ),
        unsafe_allow_html=True,
    )

    _, col_flip, _ = st.columns([1, 2, 1])
    with col_flip:
        if st.button("Lật lại" if flipped else "Xem đáp án", use_container_width=True, type="primary"):
            st.session_state.fc_flipped = not flipped
            st.rerun()

    col_prev, col_src, col_next = st.columns(3)
    with col_prev:
        if st.button("Trước", disabled=(idx == 0), use_container_width=True):
            st.session_state.fc_idx -= 1
            st.session_state.fc_flipped = False
            st.rerun()
    with col_src:
        if card.source_markers:
            st.caption("Nguồn: " + ", ".join(card.source_markers))
    with col_next:
        if st.button("Tiếp", disabled=(idx == len(cards) - 1), use_container_width=True):
            st.session_state.fc_idx += 1
            st.session_state.fc_flipped = False
            st.rerun()

    _render_citations(res.citations)
    st.divider()
    col_a, col_b = st.columns(2)
    col_a.download_button("Tải JSON", res.model_dump_json(indent=2), file_name="flashcards.json", mime="application/json")
    col_b.download_button("Tải Markdown", flashcards_to_markdown(res), file_name="flashcards.md", mime="text/markdown")


def _tab_history() -> None:
    st.subheader("Lịch sử học tập")
    if not st.session_state.chat_history and not st.session_state.history:
        st.info("Chưa có hoạt động nào trong phiên này.")
        return

    label_map = {"summary": "Tóm tắt", "quiz": "Quiz", "flashcards": "Flashcards"}

    if st.session_state.chat_history:
        st.markdown("#### Hỏi đáp")
        for msg in st.session_state.chat_history:
            role = "Bạn" if msg["role"] == "user" else "Trợ lý"
            preview = msg["content"].strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "..."
            st.markdown(f"- **{role}:** {preview}")
        if st.button("Xoá lịch sử hỏi đáp"):
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.history:
        st.markdown("#### Các tạo sinh gần đây")
        for i, h in enumerate(st.session_state.history, start=1):
            target = h.get("target") or "(toàn bộ kho)"
            label = label_map.get(h["kind"], h["kind"])
            extra = f" · {h['count']} mục" if "count" in h else ""
            st.markdown(f"{i}. **{label}** — {target}{extra}")
        if st.button("Xoá lịch sử tạo sinh"):
            st.session_state.history = []
            st.rerun()


def run() -> None:
    st.set_page_config(page_title="RAG Learning", page_icon="📚", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    _init_state()
    document, page = _sidebar()

    st.title("Hệ thống học tập với RAG")
    st.caption("Hỏi đáp, tóm tắt, quiz, và flashcards có trích dẫn nguồn — dựa trên PDF đã nạp.")

    tabs = st.tabs(["💬 Hỏi đáp", "📝 Tóm tắt", "📋 Quiz", "🃏 Flashcards", "📖 Lịch sử"])
    with tabs[0]:
        _tab_chat(document, page)
    with tabs[1]:
        _tab_summary(document, page)
    with tabs[2]:
        _tab_quiz(document, page)
    with tabs[3]:
        _tab_flashcards(document, page)
    with tabs[4]:
        _tab_history()


run()
