import streamlit as st

# ========= I18N (navigation labels) =========
NAV_LANG = {
    "en": {
        "browser_title": "Shooter Analytics",
        "select_language": "Language",
        "page_home": "Home",
        "page_match": "Match Analysis",
        "page_shooter": "Shooter Performance",
        "page_comparison": "Shooter Comparison",
    },
    "it": {
        "browser_title": "Shooter Analytics",
        "select_language": "Lingua",
        "page_home": "Home",
        "page_match": "Analisi Match",
        "page_shooter": "Prestazioni Tiratore",
        "page_comparison": "Confronto Tiratori",
    },
}


def nav_t(key: str, lang: str) -> str:
    return NAV_LANG.get(lang, NAV_LANG["en"]).get(key, NAV_LANG["en"].get(key, key))


# ========= SESSION STATE =========
if "language" not in st.session_state:
    st.session_state.language = "it"

st.set_page_config(
    page_title=nav_t("browser_title", st.session_state.language),
    layout="wide",
)

# ========= SIDEBAR (shared across all pages) =========
language_options = list(NAV_LANG.keys())
selected_lang = st.sidebar.selectbox(
    nav_t("select_language", st.session_state.language),
    options=language_options,
    index=language_options.index(st.session_state.language),
    key="dd_language_global",
)
st.session_state.language = selected_lang

st.sidebar.image("assets/ipsc-seeklogo.png")
st.sidebar.image("assets/FITDS-logo-1.png")

# ========= NAVIGATION =========
lang = st.session_state.language

home = st.Page(
    "views/home.py",
    title=nav_t("page_home", lang),
    icon="🏠",
    default=True,
)
match_analysis = st.Page(
    "views/match_analysis.py",
    title=nav_t("page_match", lang),
    icon="⛳️",
)
shooter_performance = st.Page(
    "views/shooter_performance.py",
    title=nav_t("page_shooter", lang),
    icon="🎯",
)
shooter_comparison = st.Page(
    "views/shooter_comparison.py",
    title=nav_t("page_comparison", lang),
    icon="🏆",
)

pg = st.navigation([home, match_analysis, shooter_performance, shooter_comparison])
pg.run()
