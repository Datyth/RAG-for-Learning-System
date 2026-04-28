"""Streamlit UI — pure frontend, calls FastAPI backend via HTTP.

Run with:
    uv run streamlit run src/interfaces/ui.py

Requires the API to be running:
    uv run rag-api

Override API URL with RAG_API_URL env var (default: http://localhost:8000).
All user-facing text is in Vietnamese.
"""

import html
from typing import TypeVar

import httpx
import streamlit as st
from pydantic import BaseModel

from src.config import settings
from src.export import export
from src.interfaces.styles import GLOBAL_CSS
from src.filters import MetadataFilter, filters_to_dict
from src.schemas import Citation, FlashcardSet, QuizSet, RagAnswer, RetrievedChunk, Summary

_API = settings.api_url
_ALL_PAGES = "(Tất cả trang)"
T = TypeVar("T", bound=BaseModel)


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    try:
        return httpx.request(method, f"{_API}{path}", timeout=120, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Không kết nối được API tại {_API}. Chạy `uv run rag-api` trước.\n\n{exc}")
        st.stop()


def _error_detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail", r.text))
    except ValueError:
        return r.text


def _post_model(path: str, payload: dict, model: type[T], err: str) -> T | None:
    r = _api("POST", path, json=payload)
    if r.is_error:
        st.error(f"{err}: {_error_detail(r)}")
        return None
    return model.model_validate(r.json())


def _filters_json(filenames: list[str], page: int | None) -> dict | None:
    payload: dict[str, object] = {}
    if filenames:
        payload["filenames"] = filenames
    if page is not None:
        payload["page"] = page
    return filters_to_dict(MetadataFilter.model_validate(payload))


def _init_state() -> None:
    for key, default in {"chat_history": [], "uploaded_files": set(), "doc_checks": {}}.items():
        st.session_state.setdefault(key, default)


def _downloads(res: BaseModel, stem: str) -> None:
    st.divider()
    c1, c2 = st.columns(2)
    c1.download_button("Tải JSON", export(res, fmt="json"), f"{stem}.json", "application/json")
    c2.download_button("Tải Markdown", export(res, fmt="md"), f"{stem}.md", "text/markdown")


def _render_citations(citations: list[Citation]) -> None:
    if not citations:
        return
    chips = "".join(
        f'<span class="src-chip"><b>{html.escape(c.source_marker)}</b>'
        f'<span class="src-chip-meta">{html.escape(c.filename)} tr.{c.page}'
        f"{' · ' + html.escape(c.section) if c.section else ''}</span></span>"
        for c in citations
    )
    with st.expander("Nguồn trích dẫn", expanded=False):
        st.markdown(f'<div class="src-chips">{chips}</div>', unsafe_allow_html=True)


def _render_chunks(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        return
    with st.expander("Chi tiết đoạn văn được trích", expanded=False):
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**[S{i}] {c.metadata.filename} — trang {c.metadata.page}** · điểm: {c.score:.3f}"
            )
            st.code(c.text.strip(), language="text")


def _sidebar() -> tuple[list[str], int | None]:
    st.sidebar.markdown(
        "<h2 style='margin:0 0 .1rem;font-size:1.3rem'>📚 RAG Learning</h2>"
        "<p style='margin:0 0 1rem;opacity:.55;font-size:.82rem'>Học tập thông minh với AI</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Tải tài liệu PDF")
    uploaded = st.sidebar.file_uploader(
        "Chọn một hoặc nhiều PDF", type=["pdf"], accept_multiple_files=True, key="uploader"
    )
    for f in uploaded or []:
        if f.name in st.session_state.uploaded_files:
            continue
        with st.sidebar.status(f"Đang nạp {f.name}...", expanded=False):
            r = _api("POST", "/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
            if r.is_error:
                st.sidebar.error(f"Lỗi: {_error_detail(r)}")
                continue
            info = r.json()
        st.session_state.uploaded_files.add(f.name)
        st.session_state.pop("cached_docs", None)
        st.sidebar.success(f"Đã nạp **{info['filename']}** · {info['chunks_indexed']} đoạn")

    st.sidebar.subheader("Bộ lọc tài liệu")
    if "cached_docs" not in st.session_state:
        try:
            st.session_state.cached_docs = _api("GET", "/documents").json()
        except Exception as exc:
            st.sidebar.error(f"Không đọc được tài liệu: {exc}")
            st.session_state.cached_docs = []
    docs = st.session_state.cached_docs

    doc_map = {d["filename"]: d for d in docs}
    options = list(doc_map)
    page = None

    if docs:
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Tài liệu", len(docs))
        c2.metric("Đoạn", sum(d["chunk_count"] for d in docs))

    checks: dict[str, bool] = st.session_state.doc_checks
    known = set(options)
    for k in list(checks):
        if k not in known:
            checks.pop(k, None)
            st.session_state.pop(f"doc_cb_{k}", None)
    for fn in options:
        checks.setdefault(fn, True)
        st.session_state.setdefault(f"doc_cb_{fn}", True)

    b1, b2 = st.sidebar.columns(2)
    if b1.button("Chọn tất cả", use_container_width=True):
        for fn in options:
            checks[fn] = True
            st.session_state[f"doc_cb_{fn}"] = True
        st.rerun()
    if b2.button("Bỏ chọn", use_container_width=True):
        for fn in options:
            checks[fn] = False
            st.session_state[f"doc_cb_{fn}"] = False
        st.rerun()

    box = st.sidebar.container(height=240) if len(options) > 12 else st.sidebar
    for fn in options:
        checks[fn] = box.checkbox(fn, value=bool(checks.get(fn)), key=f"doc_cb_{fn}")

    filenames = [fn for fn in options if checks.get(fn)]
    if len(filenames) == 1:
        pages = doc_map[filenames[0]]["pages"]
        chosen = st.sidebar.selectbox(
            "Trang", [_ALL_PAGES, *map(str, pages)], index=0, key="page_select"
        )
        page = None if chosen == _ALL_PAGES else int(chosen)
    elif len(filenames) > 1:
        st.sidebar.caption("Lọc theo trang không áp dụng khi chọn nhiều tài liệu.")

    return filenames, page


def _tab_chat(filenames: list[str], page: int | None) -> None:
    st.subheader("Hỏi đáp có trích dẫn")
    st.caption("Đặt câu hỏi về nội dung đã nạp. Bộ lọc ở thanh bên sẽ thu hẹp phạm vi tìm kiếm.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_citations([Citation(**c) for c in msg.get("citations", [])])
            _render_chunks([RetrievedChunk(**c) for c in msg.get("chunks", [])])

    q = st.chat_input("Đặt câu hỏi về tài liệu...")
    if not q:
        return

    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            res = _post_model(
                "/ask",
                {"question": q, "filters": _filters_json(filenames, page)},
                RagAnswer,
                "Không thể trả lời",
            )
        if not res:
            return
        st.markdown(res.answer)
        _render_citations(res.citations)
        _render_chunks(res.chunks)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": res.answer,
            "citations": [c.model_dump() for c in res.citations],
            "chunks": [c.model_dump() for c in res.chunks],
        }
    )


def _tab_summary(filenames: list[str], page: int | None) -> None:
    st.subheader("Tóm tắt tài liệu")
    with st.form("summary_form"):
        query = st.text_input("Chủ đề hướng dẫn (tuỳ chọn)")
        k = st.number_input("Số đoạn truy xuất (k)", 1, 64, 12)
        submit = st.form_submit_button("Tạo tóm tắt", use_container_width=True, type="primary")
    if not submit:
        return

    with st.spinner("Đang tóm tắt..."):
        res = _post_model(
            "/summarize",
            {
                "document": None,
                "query": query or None,
                "filters": _filters_json(filenames, page),
                "k": int(k),
            },
            Summary,
            "Không thể tóm tắt",
        )
    if not res:
        return
    if not res.summary:
        st.warning("Không tìm thấy nội dung phù hợp để tóm tắt.")
        return

    scope = {
        "query": "Theo chủ đề",
        "document": "Theo tài liệu",
        "filter": "Theo bộ lọc",
        "corpus": "Toàn bộ kho",
    }.get(res.scope, res.scope)
    st.caption(f"Phạm vi: {scope}" + (f" · {res.target}" if res.target else ""))
    st.markdown(res.summary)

    if res.key_points:
        items = "".join(f"<li>{html.escape(kp)}</li>" for kp in res.key_points)
        st.markdown(
            f"<p style='font-weight:600;margin:.75rem 0 .25rem'>Các ý chính</p><ul class='kp-list'>{items}</ul>",
            unsafe_allow_html=True,
        )

    _render_citations(res.citations)
    _downloads(res, "summary")


def _clear_quiz_state() -> None:
    for k in [k for k in st.session_state if k.startswith(("quiz_q_", "quiz_ans_"))]:
        del st.session_state[k]


def _tab_quiz(filenames: list[str], page: int | None) -> None:
    st.subheader("Tạo bộ câu hỏi")

    with st.expander("Tạo quiz mới", expanded="quiz_res" not in st.session_state):
        with st.form("quiz_form"):
            query = st.text_input("Chủ đề (tuỳ chọn)")
            c1, c2 = st.columns(2)
            count = c1.number_input("Số câu hỏi", 1, 30, 8)
            k = c2.number_input("Số đoạn truy xuất", 1, 64, 16)
            submit = st.form_submit_button("Tạo quiz", use_container_width=True, type="primary")
        if submit:
            with st.spinner("Đang tạo quiz..."):
                res = _post_model(
                    "/quiz",
                    {
                        "document": None,
                        "query": query or None,
                        "filters": _filters_json(filenames, page),
                        "count": int(count),
                        "k": int(k),
                    },
                    QuizSet,
                    "Không thể tạo quiz",
                )
            if res:
                st.session_state.quiz_res = res
                st.session_state.quiz_submitted = False
                _clear_quiz_state()

    if "quiz_res" not in st.session_state:
        return

    res: QuizSet = st.session_state.quiz_res
    submitted = st.session_state.get("quiz_submitted", False)

    if submitted:
        correct = sum(
            st.session_state.get(f"quiz_ans_{i}") == item.correct_index
            for i, item in enumerate(res.items)
        )
        pct = int(100 * correct / len(res.items))
        cls = "score-high" if pct >= 70 else "score-mid" if pct >= 40 else "score-low"
        col_score, col_btn = st.columns([3, 1])

        with col_score:
            st.markdown(
                f'<div class="score-wrap"><span class="score-badge {cls}">{correct}/{len(res.items)}</span>'
                f'<span class="score-label">{pct}% chính xác</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(pct / 100)
        with col_btn:
            st.write("")
            if st.button("Làm lại", use_container_width=True):
                st.session_state.quiz_submitted = False
                _clear_quiz_state()
                st.rerun()

    for i, item in enumerate(res.items):
        meta = [x for x in [item.topic, item.difficulty] if x]
        suffix = f" · _{' · '.join(meta)}_" if meta else ""

        with st.container(border=True):
            st.markdown(f"**Câu {i + 1}.**{suffix}")
            st.markdown(item.question)
            st.radio(
                f"q{i}",
                list(range(len(item.options))),
                key=f"quiz_q_{i}",
                format_func=lambda j, opts=item.options: f"{chr(65 + j)}. {opts[j]}",
                label_visibility="collapsed",
                index=None,
                disabled=submitted,
            )

            if submitted:
                ans = st.session_state.get(f"quiz_ans_{i}")
                correct_label = (
                    f"**{chr(65 + item.correct_index)}. {item.options[item.correct_index]}**"
                )
                if ans == item.correct_index:
                    st.success(f"Đúng! Đáp án: {correct_label}")
                elif ans is None:
                    st.warning(f"Chưa chọn. Đáp án: {correct_label}")
                else:
                    st.error(f"Sai. Đáp án đúng: {correct_label}")
                if item.explanation:
                    with st.expander("Giải thích"):
                        st.markdown(item.explanation)
            elif item.source_markers:
                st.caption("Nguồn: " + ", ".join(item.source_markers))

    if not submitted and st.button("Nộp bài", type="primary"):
        for i in range(len(res.items)):
            st.session_state[f"quiz_ans_{i}"] = st.session_state.get(f"quiz_q_{i}")
        st.session_state.quiz_submitted = True
        st.rerun()

    _render_citations(res.citations)
    _downloads(res, "quiz")


def _fc_card_html(face: str, side: str, topic: str | None, hint: str | None, flipped: bool) -> str:
    topic_html = f'<p class="fc-topic">{html.escape(topic)}</p>' if topic and not flipped else ""
    hint_html = f'<p class="fc-hint">{html.escape(hint)}</p>' if hint and flipped else ""
    return (
        f'<div class="fc-card {"fc-back" if flipped else "fc-front"}"><p class="fc-side">{side}</p>{topic_html}'
        f'<p class="fc-text">{html.escape(face).replace(chr(10), "<br>")}</p>{hint_html}</div>'
    )


def _tab_flashcards(filenames: list[str], page: int | None) -> None:
    st.subheader("Tạo thẻ ghi nhớ")

    with st.expander("Tạo bộ thẻ mới", expanded="fc_res" not in st.session_state):
        with st.form("fc_form"):
            query = st.text_input("Chủ đề (tuỳ chọn)")
            c1, c2 = st.columns(2)
            count = c1.number_input("Số thẻ", 1, 40, 15)
            k = c2.number_input("Số đoạn truy xuất", 1, 64, 16)
            submit = st.form_submit_button("Tạo thẻ", use_container_width=True, type="primary")
        if submit:
            with st.spinner("Đang tạo flashcards..."):
                res = _post_model(
                    "/flashcards",
                    {
                        "document": None,
                        "query": query or None,
                        "filters": _filters_json(filenames, page),
                        "count": int(count),
                        "k": int(k),
                    },
                    FlashcardSet,
                    "Không thể tạo thẻ",
                )
            if res:
                st.session_state.fc_res = res
                st.session_state.fc_idx = 0
                st.session_state.fc_flipped = False

    if "fc_res" not in st.session_state:
        return

    res: FlashcardSet = st.session_state.fc_res
    cards = res.cards
    if not cards:
        st.warning("Không có thẻ nào được tạo.")
        return

    idx = min(st.session_state.get("fc_idx", 0), len(cards) - 1)
    flipped = st.session_state.get("fc_flipped", False)
    card = cards[idx]

    st.progress((idx + 1) / len(cards), text=f"Thẻ {idx + 1} / {len(cards)}")
    st.markdown(
        _fc_card_html(
            card.back if flipped else card.front,
            "Mặt sau" if flipped else "Mặt trước",
            card.topic,
            card.hint,
            flipped,
        ),
        unsafe_allow_html=True,
    )

    _, col_flip, _ = st.columns([1, 2, 1])
    with col_flip:
        if st.button(
            "Lật lại" if flipped else "Xem đáp án", use_container_width=True, type="primary"
        ):
            st.session_state.fc_flipped = not flipped
            st.rerun()

    col_prev, col_src, col_next = st.columns(3)
    with col_prev:
        if st.button("Trước", disabled=idx == 0, use_container_width=True):
            st.session_state.fc_idx = idx - 1
            st.session_state.fc_flipped = False
            st.rerun()
    with col_src:
        if card.source_markers:
            st.caption("Nguồn: " + ", ".join(card.source_markers))
    with col_next:
        if st.button("Tiếp", disabled=idx == len(cards) - 1, use_container_width=True):
            st.session_state.fc_idx = idx + 1
            st.session_state.fc_flipped = False
            st.rerun()

    _render_citations(res.citations)
    _downloads(res, "flashcards")


def run() -> None:
    st.set_page_config(page_title="RAG Learning", page_icon="📚", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    _init_state()
    filenames, page = _sidebar()

    col_title, col_ctx = st.columns([3, 1])
    with col_title:
        st.markdown(
            "<h1 style='margin-bottom:.1rem'>Hệ thống học tập với RAG</h1>"
            "<p style='margin:0 0 1rem;opacity:.6'>Hỏi đáp · Tóm tắt · Quiz · Flashcards — có trích dẫn nguồn từ PDF</p>",
            unsafe_allow_html=True,
        )
    with col_ctx:
        if not filenames:
            st.info("📚 Toàn bộ kho tài liệu", icon=None)
        elif len(filenames) == 1:
            st.info(f"📄 {filenames[0]}" + (f"\n\nTrang {page}" if page else ""), icon=None)
        else:
            st.info(f"📄 {len(filenames)} tài liệu đã chọn", icon=None)

    tabs = st.tabs(["💬 Hỏi đáp", "📝 Tóm tắt", "📋 Quiz", "🃏 Flashcards"])
    for tab, fn in zip(tabs, [_tab_chat, _tab_summary, _tab_quiz, _tab_flashcards]):
        with tab:
            fn(filenames, page)


run()
