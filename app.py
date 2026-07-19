"""Module 3 - The Decision Report. The Genome Firewall demo app.

A Streamlit app that takes a genome, runs the calibrated per-antibiotic models,
and returns for each drug: likely to work / likely to fail / no-call, a calibrated
confidence, an evidence category, and an evidence panel linking the genes behind
the call to CARD and NCBI. Every result carries the mandatory reminder that it
must be confirmed by standard laboratory testing.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config, data_sim, evidence, features, metrics
from train import build

st.set_page_config(page_title="Genome Firewall", page_icon="🧬", layout="wide")

# --- colours for each call ---------------------------------------------------
CALL_STYLE = {
    "likely to work": ("✅", "#1a7f37", "Likely to WORK"),
    "likely to fail": ("⛔", "#b42318", "Likely to FAIL"),
    "no-call":        ("⚠️", "#b54708", "NO-CALL (confirm in lab)"),
}


@st.cache_resource
def get_bundle():
    """Train in memory the first time, then cache. Returns model, data and report."""
    model, report = build(save=False)
    X, y, meta = data_sim.generate()
    return model, X, y, meta, report


def call_badge(call: str) -> str:
    icon, colour, label = CALL_STYLE.get(call, ("", "#333", call))
    return (f"<span style='background:{colour};color:white;padding:3px 10px;"
            f"border-radius:12px;font-weight:600'>{icon} {label}</span>")


def feature_row_for_example(X: pd.DataFrame, meta: pd.DataFrame, genome_id: str) -> pd.Series:
    idx = meta.index[meta["genome_id"] == genome_id][0]
    return X.iloc[idx]


# Mechanism icons, so the biology reads at a glance.
MECH_ICON = {
    "Enzymatic degradation": "✂️",
    "Target modification": "🔧",
    "Target protection": "🛡️",
    "Drug modification": "🏷️",
}

# The pipeline as a picture (rendered in the About tab).
ARCH_DOT = """digraph {
    rankdir=LR; bgcolor="transparent";
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#cbd5e1"];
    genome [label="Genome (FASTA)", fillcolor="#eef2ff"];
    feat   [label="AMRFinderPlus\\nfeatures", fillcolor="#eef2ff"];
    model  [label="Calibrated model\\nper antibiotic", fillcolor="#eef2ff"];
    call   [label="work / fail / no-call\\n+ confidence", fillcolor="#dcfce7"];
    evid   [label="Evidence\\ngenes, CARD, NCBI", fillcolor="#fef3c7"];
    report [label="Decision report", fillcolor="#fee2e2"];
    genome -> feat -> model -> call -> report;
    evid -> report [style=dashed, label="explains"];
}"""


def antibiogram_figure(results):
    """A one-glance bar chart, predicted chance each drug fails, coloured by call."""
    import matplotlib.pyplot as plt

    colour_map = {"likely to work": "#1a7f37", "likely to fail": "#b42318", "no-call": "#b54708"}
    labels, vals, colours = [], [], []
    for abx in config.ANTIBIOTICS:
        r = results[abx]
        p = r["probability_resistant"]
        labels.append(abx)
        vals.append(100.0 if p is None else p * 100.0)
        colours.append(colour_map.get(r["call"], "#888888"))

    fig, ax = plt.subplots(figsize=(7, 2.7))
    y = list(range(len(labels)))
    ax.barh(y, vals, color=colours)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.axvline(50, color="grey", ls="--", lw=1)
    ax.set_xlabel("Predicted chance the drug fails (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    for i, v in enumerate(vals):
        ax.text(min(v + 1.5, 96), i, f"{v:.0f}%", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def mechanism_dot(gene, protein, mechanism):
    """Gene to protein to mechanism to resistance, as a small left-to-right diagram."""
    return (
        'digraph { rankdir=LR; bgcolor="transparent"; '
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#cbd5e1"]; '
        f'g [label="{gene}\\n(gene)", fillcolor="#eef2ff"]; '
        f'p [label="{protein}\\n(protein)", fillcolor="#eef2ff"]; '
        f'm [label="{mechanism}", fillcolor="#fef3c7"]; '
        'r [label="Resistance", fillcolor="#fee2e2"]; '
        'g -> p -> m -> r; }'
    )


def render_bucket(col, title, colour, items):
    """One colour-coded triage column with the drugs that fall into it."""
    col.markdown(
        f"<div style='background:{colour};color:white;padding:6px 10px;border-radius:8px;"
        f"font-weight:600;text-align:center'>{title}</div>",
        unsafe_allow_html=True,
    )
    if not items:
        col.caption("—")
    for abx, r in items:
        brand = config.BRAND_NAMES.get(abx, "")
        conf = "" if r["probability_resistant"] is None else f" · {r['confidence']*100:.0f}%"
        col.markdown(
            f"<div style='padding:8px 4px 0'><b>{abx}</b>"
            f"<br><span style='color:#6b7280;font-size:0.82em'>{brand}{conf}</span></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
model, X, y, meta, report = get_bundle()

st.title("🧬 Genome Firewall")
st.caption(f"A defensive AI decision-support tool for antibiotic resistance · {config.SPECIES}")

st.error(
    "⚠️ Research prototype. Every result below is decision support only and MUST be "
    "confirmed by standard laboratory testing before any treatment decision. This tool "
    "predicts and explains existing resistance. It never designs or modifies organisms."
)

with st.sidebar:
    st.header("Input genome")
    mode = st.radio("Choose a source", ["Example genome", "Upload FASTA"])

    use_llm = st.toggle("Plain-language notes via OpenAI", value=False,
                        help="Optional. Needs OPENAI_API_KEY. The LLM only rephrases "
                             "curated facts, it never changes the prediction.")

    selected_row = None
    if mode == "Example genome":
        ids = meta["genome_id"].head(25).tolist()
        genome_id = st.selectbox("Example genome", ids)
        selected_row = feature_row_for_example(X, meta, genome_id)
    else:
        upload = st.file_uploader("Assembled genome (.fasta / .fna)", type=["fasta", "fna", "fa"])
        if upload is not None:
            if features.amrfinder_available():
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp:
                    tmp.write(upload.read())
                    tmp_path = tmp.name
                genes = features.run_amrfinderplus(tmp_path)
                selected_row = features.genes_to_feature_row(genes)
                st.success(f"AMRFinderPlus detected {len(genes)} elements.")
            else:
                st.info("AMRFinderPlus is not installed here, so this deployed demo uses "
                        "an example feature profile instead. Locally, install it from "
                        "github.com/ncbi/amr for real FASTA annotation.")
                selected_row = feature_row_for_example(X, meta, meta["genome_id"].iloc[0])

tab_predict, tab_validation, tab_about = st.tabs(["Prediction", "Model validation", "About & safety"])

# --- Prediction tab ----------------------------------------------------------
with tab_predict:
    if selected_row is None:
        st.info("Choose an example genome or upload a FASTA in the sidebar.")
    else:
        present = [g for g in config.RESISTANCE_GENES if int(selected_row.get(g, 0)) == 1]
        st.markdown(f"**Resistance genes detected in this genome:** "
                    f"{', '.join(present) if present else 'none'}")

        results = model.predict_one(selected_row)

        st.pyplot(antibiogram_figure(results))
        st.caption("Bar shows the predicted chance the drug fails. Green works, red fails, amber no-call.")
        st.divider()

        # Triage board, drugs sorted into three colour-coded columns (like a
        # stewardship tool), so the prescriber sees the shortlist at a glance.
        buckets = {"work": [], "no-call": [], "fail": []}
        for abx in config.ANTIBIOTICS:
            key = {"likely to work": "work", "likely to fail": "fail"}.get(results[abx]["call"], "no-call")
            buckets[key].append((abx, results[abx]))

        cols = st.columns(3)
        render_bucket(cols[0], "🟢 Recommended", "#1a7f37", buckets["work"])
        render_bucket(cols[1], "🟡 No-call, confirm in lab", "#b54708", buckets["no-call"])
        render_bucket(cols[2], "🔴 Likely to fail", "#b42318", buckets["fail"])

        st.divider()
        st.markdown("**Evidence and mechanism** — open a drug for its genes, mechanism and sources.")
        for abx in config.ANTIBIOTICS:
            r = results[abx]
            brand = config.BRAND_NAMES.get(abx)
            label = abx + (f" ({brand})" if brand else "") + f"  —  {r['call']}"
            with st.expander(label):
                st.write(r["reason"])
                panel = evidence.build_evidence_panel(abx, r["driver_genes"], use_llm=use_llm)
                if panel:
                    for item in panel:
                        icon = MECH_ICON.get(item["mechanism"], "🧬")
                        st.markdown(f"**{item['gene']}**  ·  {icon} *{item['mechanism']}*")
                        st.markdown(item["note"])
                        st.graphviz_chart(mechanism_dot(item["gene"], item["protein"], item["mechanism"]))
                        st.markdown(
                            f"[CARD]({item['card_url']}) · [NCBI]({item['ncbi_url']}) · "
                            f"[Literature]({item['pubmed_url']}) · [3D structure]({item['structure_url']})"
                        )
                elif r["evidence_category"] == evidence.STATISTICAL:
                    st.markdown(
                        "_Statistical association only. No known gene or mechanism, so no biological "
                        "cause is claimed. Confirm in the lab._"
                    )
                else:
                    st.markdown("_No known resistance determinant for this drug in this genome._")

        st.warning("Confirm every result with standard laboratory testing before treating.")

# --- Validation tab ----------------------------------------------------------
with tab_validation:
    st.subheader("Performance on held-out lineages (grouped split)")
    st.caption("Whole lineages are held out for testing, so the model is scored on genetically "
               "distinct genomes it never saw. This is the honest test of generalisation.")

    rows = []
    for abx, r in report.items():
        m = r["metrics"]
        rows.append({
            "Antibiotic": abx,
            "Balanced acc": round(m["balanced_accuracy"], 3),
            "Recall (R)": round(m["recall_resistant"], 3),
            "Recall (S)": round(m["recall_susceptible"], 3),
            "AUROC": round(m["auroc"], 3),
            "PR-AUC": round(m["pr_auc"], 3),
            "Brier": round(m["brier"], 3),
            "No-call rate": round(m["no_call_rate"], 2),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Calibration (reliability curves)")
    try:
        import matplotlib.pyplot as plt

        cols = st.columns(len(config.ANTIBIOTICS))
        for col, abx in zip(cols, config.ANTIBIOTICS):
            mean_pred, obs_frac, _ = report[abx]["reliability"]
            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            ax.plot([0, 1], [0, 1], "--", color="grey")
            ax.plot(mean_pred, obs_frac, "o-")
            ax.set_title(abx, fontsize=8)
            ax.set_xlabel("predicted", fontsize=7)
            ax.set_ylabel("observed", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.tight_layout()
            col.pyplot(fig)
    except ImportError:
        st.info("Install matplotlib to see reliability plots.")

# --- About tab ---------------------------------------------------------------
with tab_about:
    st.subheader("What this is")
    st.write("Genome Firewall turns a bacterial genome into an earlier, evidence-based "
             "prediction of which antibiotics are likely to work, before slow cultures return.")

    st.subheader("How it is built")
    st.graphviz_chart(ARCH_DOT)
    st.caption("Genome to AMRFinderPlus features to a calibrated model per antibiotic to the "
               "report. Evidence only explains the call, it never changes it.")

    st.subheader("Strictly defensive")
    st.write("It predicts and explains resistance that already exists. It never designs, "
             "modifies, or optimises an organism, and every report must be confirmed by a "
             "trained professional and by standard laboratory testing.")
