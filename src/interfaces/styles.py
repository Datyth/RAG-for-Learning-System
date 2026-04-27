"""Custom CSS injected into the Streamlit app at startup."""

GLOBAL_CSS = """<style>
/* ── Design tokens ── */
:root {
    --accent:        #4f46e5;
    --accent-light:  rgba(79,70,229,.08);
    --accent-border: rgba(79,70,229,.28);
    --radius:        .875rem;
    --radius-lg:     1.25rem;
    --shadow:        0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.05);
    --shadow-md:     0 4px 12px rgba(0,0,0,.08),0 2px 4px rgba(0,0,0,.05);
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size:.95rem !important;
    font-weight:600 !important;
    padding:.55rem 1.1rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color:var(--accent) !important; }

/* ── Bordered containers ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius:var(--radius) !important;
    box-shadow:var(--shadow) !important;
    border-color:rgba(128,128,128,.15) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { padding:1.25rem 1.5rem !important; }

/* ── Download buttons ── */
.stDownloadButton button { width:100%; }

/* ── Citation chips ── */
.src-chips { display:flex; flex-wrap:wrap; gap:.35rem; margin:.25rem 0 0; }
.src-chip {
    display:inline-flex; align-items:center; gap:.25rem;
    background:var(--accent-light); border:1px solid var(--accent-border);
    border-radius:2rem; padding:.18rem .6rem;
    font-size:.78rem; font-weight:600; color:var(--accent); line-height:1.5;
}
.src-chip-meta { opacity:.6; font-weight:400; }

/* ── Key-points list ── */
.kp-list { list-style:none; padding:0; margin:.5rem 0 0; }
.kp-list li {
    display:flex; gap:.6rem; align-items:baseline;
    padding:.3rem 0; font-size:.95rem; line-height:1.6;
    border-bottom:1px solid rgba(128,128,128,.08);
}
.kp-list li:last-child { border-bottom:none; }
.kp-list li::before { content:"✦"; color:var(--accent); font-size:.6rem; flex-shrink:0; margin-top:.25rem; }

/* ── Score badge ── */
.score-wrap { display:flex; align-items:center; gap:.75rem; margin:.25rem 0 .75rem; }
.score-badge {
    display:inline-block; padding:.3rem .85rem;
    border-radius:2rem; font-weight:700; font-size:.9rem;
}
.score-high { background:rgba(22,163,74,.1);  color:#15803d; border:1px solid rgba(22,163,74,.3); }
.score-mid  { background:rgba(202,138,4,.1);  color:#a16207; border:1px solid rgba(202,138,4,.3); }
.score-low  { background:rgba(220,38,38,.1);  color:#b91c1c; border:1px solid rgba(220,38,38,.3); }
.score-label { font-size:.9rem; opacity:.7; }

/* ── Flashcard ── */
.fc-card {
    border-radius:var(--radius-lg); padding:2.75rem 2.5rem; min-height:220px;
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; text-align:center;
    margin-bottom:1.25rem; box-shadow:var(--shadow-md); transition:box-shadow .2s;
}
.fc-front { background:var(--secondary-background-color,#f8f9fa); border:1.5px solid rgba(128,128,128,.15); }
.fc-back  { background:var(--accent-light); border:1.5px solid var(--accent-border); }
.fc-side  { font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; opacity:.38; margin:0 0 .5rem; }
.fc-topic { font-size:.78rem; font-weight:600; color:var(--accent); opacity:.9; margin:0 0 .85rem; }
.fc-text  { font-size:1.18rem; font-weight:500; line-height:1.8; margin:0; }
.fc-hint  { font-size:.83rem; opacity:.55; margin:.8rem 0 0; font-style:italic; }

/* ── Chat input ── */
textarea[data-testid="stChatInputTextArea"] { font-size:1rem !important; }
</style>"""
