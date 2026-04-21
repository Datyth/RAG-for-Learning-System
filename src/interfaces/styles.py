"""Custom CSS injected into the Streamlit app at startup."""

GLOBAL_CSS = """<style>
/* Tabs: larger click targets and bolder labels */
button[data-baseweb="tab"] { font-size:1rem !important; font-weight:600 !important; padding:.6rem 1.2rem !important; }

/* Bordered cards: more breathing room and softer corners */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:.75rem !important; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { padding:1.25rem 1.5rem !important; }

/* Chat input: match body font size */
textarea[data-testid="stChatInputTextArea"] { font-size:1rem !important; }

/* Download buttons: stretch to full column width */
.stDownloadButton button { width:100%; }

/* ── Flashcard faces ── */
.fc-card {
    border-radius:1rem;
    padding:2.5rem 2rem;
    min-height:200px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    text-align:center;
    border:1.5px solid rgba(128,128,128,.2);
    margin-bottom:1.25rem;
    transition:border-color .2s;
}
.fc-front { background:var(--secondary-background-color,#f8f9fa); }
.fc-back  { border-color:rgba(49,130,206,.5); background:rgba(49,130,206,.06); }
.fc-side  { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; opacity:.45; margin:0 0 .4rem; }
.fc-topic { font-size:.8rem; font-weight:600; opacity:.65; margin:0 0 .75rem; }
.fc-text  { font-size:1.15rem; font-weight:500; line-height:1.75; margin:0; }
.fc-hint  { font-size:.85rem; opacity:.6; margin:.75rem 0 0; font-style:italic; }
</style>"""
