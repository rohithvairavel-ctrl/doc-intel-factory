"""Optional ops dashboard for the document-intelligence factory."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Doc Intel Factory", layout="wide")
st.title("Document Intelligence Factory — Ops Dashboard")
st.caption("Synthetic corpus · hybrid extraction · RAG eval · active learning · feedback OPE")

art = Path("artifacts/factory_results.json")
if not art.exists():
    st.warning("No artifacts/factory_results.json yet. Run: `doc-intel-factory run`")
    st.stop()

data = json.loads(art.read_text(encoding="utf-8"))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", data["corpus"]["n"])
c2.metric("F1 lift (A)", f"{data['case_a_extraction']['f1_lift']:.3f}")
c3.metric("Faithfulness (B)", f"{data['case_b_rag']['faithfulness']:.3f}")
c4.metric("OPE IPS |err|", f"{data['case_c_ope']['ips_abs_error']:.4f}")

st.subheader("Active learning curve")
al = data["case_c_active_learning"]
st.line_chart(
    {
        "uncertainty": [p["f1_uncertainty"] for p in al],
        "random": [p["f1_random"] for p in al],
    }
)

st.subheader("OPE comparison")
ope = data["case_c_ope"]
st.table(
    {
        "estimator": ["IPS", "SNIPS", "Naive", "Oracle"],
        "value": [
            ope["ips"]["estimate"],
            ope["snips"]["estimate"],
            ope["naive"]["estimate"],
            ope["oracle_target_value"],
        ],
        "abs_error": [
            ope["ips_abs_error"],
            ope["snips_abs_error"],
            ope["naive_abs_error"],
            0.0,
        ],
    }
)

st.subheader("Report")
report = Path("artifacts/factory_report.md")
if report.exists():
    st.markdown(report.read_text(encoding="utf-8"))
