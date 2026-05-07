import streamlit as st

# ========= I18N =========
LANG = {
    "en": {
        "title": "Shooter Analytics Dashboard",
        "subtitle": "Performance analytics for IPSC competitive shooting matches.",
        "nav_hint": "👈 Use the sidebar to navigate between views.",
        "intro_header": "About this app",
        "intro_text": (
            "This dashboard analyses results from IPSC matches sourced from the FITDS "
            "scoring exports. It turns raw stage-by-stage scores into clear, comparable "
            "metrics so shooters can track progress, study individual matches, and "
            "benchmark themselves against others in the same division."
        ),
        "pages_header": "Pages",
        "page_match_title": "⛳️ Match Analysis",
        "page_match_text": (
            "Drill into a single match: standings by division and class, stage-level "
            "results, distributions of points and times, and the shape of the field "
            "you competed against."
        ),
        "page_shooter_title": "🎯 Shooter Performance",
        "page_shooter_text": (
            "All-time view of one shooter: aggregated metrics (mean result, % pts, "
            "% time), match history, stage points/time scatter, and per-stage "
            "performance distributions across the matches you select."
        ),
        "page_comparison_title": "🏆 Shooter Comparison",
        "page_comparison_text": (
            "Compare up to three shooters in the same division side by side: "
            "aggregated metrics dashboard, match-level results trend, and stage "
            "points / time percentage trends over time."
        ),
        "metrics_header": "How the metrics are computed",
        "metric_result": (
            "**Mean result** — average match-level division percentage "
            "(your match points / division winner's match points), averaged across "
            "the selected matches."
        ),
        "metric_pts": (
            "**% pts** — total stage points obtained divided by total available "
            "stage points (`sum(pts) / sum(stg_max_pts)`) across the selected stages. "
            "Absolute, not division-normalised."
        ),
        "metric_time": (
            "**% time** — average division time percentage at stage level "
            "(`fastest_time_in_division / your_time`); higher means faster."
        ),
        "data_header": "Data",
        "data_text": (
            "Match data comes from FITDS PDF exports parsed into a unified CSV. "
            "Filters in each page (year, championship, match level, division, "
            "specific matches) cascade into every chart and metric on that page."
        ),
        "lang_note": "All pages support Italian and English — switch from the sidebar.",
    },
    "it": {
        "title": "Shooter Analytics Dashboard",
        "subtitle": "Analisi delle prestazioni nelle gare di tiro dinamico IPSC.",
        "nav_hint": "👈 Usa la barra laterale per navigare tra le viste.",
        "intro_header": "L'applicazione",
        "intro_text": (
            "Questa dashboard analizza i risultati delle gare IPSC a partire dagli "
            "export di classifica FITDS. Trasforma i punteggi stage per stage in "
            "metriche chiare e confrontabili, così da seguire i propri progressi, "
            "studiare le singole gare e confrontarsi con altri tiratori della stessa "
            "divisione."
        ),
        "pages_header": "Pagine",
        "page_match_title": "⛳️ Analisi Match",
        "page_match_text": (
            "Approfondimento su una singola gara: classifiche per divisione e classe, "
            "risultati a livello di stage, distribuzioni di punti e tempi, e "
            "fotografia del campo gara contro cui hai gareggiato."
        ),
        "page_shooter_title": "🎯 Prestazioni Tiratore",
        "page_shooter_text": (
            "Vista storica di un singolo tiratore: metriche aggregate (risultato "
            "medio, % punti, % tempo), storico match, scatter punti/tempo per stage "
            "e distribuzioni delle prestazioni stage per stage sui match selezionati."
        ),
        "page_comparison_title": "🏆 Confronto Tiratori",
        "page_comparison_text": (
            "Confronto fino a tre tiratori nella stessa divisione: dashboard di "
            "metriche aggregate, andamento dei risultati a livello match, e trend "
            "delle percentuali di punti e tempo a livello stage nel tempo."
        ),
        "metrics_header": "Come vengono calcolate le metriche",
        "metric_result": (
            "**Risultato medio** — media della percentuale di divisione a livello "
            "match (punti match del tiratore / punti del vincitore di divisione), "
            "calcolata sui match selezionati."
        ),
        "metric_pts": (
            "**% punti** — somma dei punti stage ottenuti divisa per la somma dei "
            "punti disponibili (`sum(pts) / sum(stg_max_pts)`) sugli stage "
            "selezionati. Valore assoluto, non normalizzato per divisione."
        ),
        "metric_time": (
            "**% tempo** — media della percentuale di tempo di divisione a livello "
            "stage (`tempo più veloce in divisione / tempo del tiratore`); più alto "
            "significa più veloce."
        ),
        "data_header": "Dati",
        "data_text": (
            "I dati delle gare provengono dai PDF di classifica FITDS, processati in "
            "un CSV unificato. I filtri di ogni pagina (anno, campionato, livello "
            "match, divisione, match specifici) si propagano a tutti i grafici e a "
            "tutte le metriche della pagina."
        ),
        "lang_note": "Tutte le pagine supportano italiano e inglese — cambia dalla barra laterale.",
    },
}


def t(key: str, lang: str) -> str:
    return LANG.get(lang, LANG["en"]).get(key, LANG["en"].get(key, key))


_ = lambda k: t(k, st.session_state.language)

st.title(_("title"))
st.write(_("subtitle"))
st.info(_("nav_hint"))

st.subheader(_("intro_header"))
st.write(_("intro_text"))

st.subheader(_("pages_header"))
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f"### {_('page_match_title')}")
    st.write(_("page_match_text"))
with p2:
    st.markdown(f"### {_('page_shooter_title')}")
    st.write(_("page_shooter_text"))
with p3:
    st.markdown(f"### {_('page_comparison_title')}")
    st.write(_("page_comparison_text"))

st.subheader(_("metrics_header"))
st.markdown(f"- {_('metric_result')}")
st.markdown(f"- {_('metric_pts')}")
st.markdown(f"- {_('metric_time')}")

st.subheader(_("data_header"))
st.write(_("data_text"))

st.caption(_("lang_note"))
