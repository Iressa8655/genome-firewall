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

        for abx in config.ANTIBIOTICS:
            r = results[abx]
            conf = "n/a" if r["probability_resistant"] is None else f"{r['confidence']*100:.0f}%"
            c1, c2, c3 = st.columns([2, 2, 6])
            c1.markdown(f"**{abx}**")
            c2.markdown(call_badge(r["call"]), unsafe_allow_html=True)
            c3.markdown(f"Confidence {conf} · evidence: *{r['evidence_category'].replace('_', ' ')}*")

            with st.expander(f"Why? — {abx}"):
                st.write(r["reason"])
                panel = evidence.build_evidence_panel(abx, r["driver_genes"], use_llm=use_llm)
                if panel:
                    for item in panel:
                        st.markdown(f"**{item['gene']}**  ·  *{item['mechanism']}*")
                        st.markdown(item["note"])
                        st.caption(
                            f"Mechanism chain:  `{item['gene']}`  →  {item['protein']}  "
                            f"→  {item['mechanism'].lower()}  →  resistance"
                        )
                        st.markdown(
                            f"[CARD]({item['card_url']}) · [NCBI]({item['ncbi_url']}) · "
                            f"[Literature]({item['pubmed_url']}) · [3D structure]({item['structure_url']})"
                        )
                elif r["evidence_category"] == evidence.STATISTICAL:
                    st.markdown(
                        "_This call rests on a **statistical association**. No known resistance gene "
                        "or mechanism was identified, so no biological cause can be claimed. Treat with "
                        "extra caution and confirm in the lab._"
                    )
                else:
                    st.markdown("_No known resistance determinant for this drug in this genome._")
            st.divider()

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
    st.markdown(
        """
### What this is
Genome Firewall turns a reconstructed bacterial genome into an earlier, evidence-based
prediction of which antibiotics are likely to work, so a care team can act before slow
culture results arrive.

### How it is built
1. **Genome Reader** — AMRFinderPlus turns the genome into resistance gene / mutation features.
2. **Predictor** — one calibrated logistic regression per antibiotic, with a deterministic
   drug-target gate and a no-call rule.
3. **Decision Report** — this app, with calibrated confidence and an evidence panel.

### The safety line
This system is **strictly defensive**. It predicts and explains resistance that already
exists to support treatment choices and public-health tracking. It never designs, modifies,
or optimises an organism. Every report must be confirmed by a trained professional and by
standard laboratory testing.
        """
    )
