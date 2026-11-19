"""
LabLens — Medical Laboratory Report Interpreter & Pre-Analytical Safety Assistant

Educational support only. It does not diagnose, validate a specimen, or replace
a qualified laboratory professional. Uploaded content is sent to Google Gemini
for processing; do not upload a report unless you have permission to do so.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# App configuration and visual style
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LabLens | Report & Sample Safety",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main { background: #f7f9fc; }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }
      .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(115deg,#102a43,#176b87);
        color: white; margin-bottom: 1rem;
      }
      .hero h1 { color: white; margin: 0 0 .3rem 0; }
      .hero p { color: #e7f5fb; margin: 0; }
      .muted { color: #62748a; font-size: .92rem; }
      .safety {
        background: #fff8e6; border-left: 5px solid #e3a008;
        padding: .9rem 1rem; border-radius: 8px; color: #493b16;
      }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5eaf1;
        padding: 12px 15px; border-radius: 12px;
      }
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] {
        background: white; border-radius: 9px 9px 0 0; padding: 10px 16px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧪 LabLens</h1>
      <p>Understand your laboratory report. Ask better questions. Support safer sample collection.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="safety"><b>Important:</b> LabLens is an educational support tool, '
    'not a diagnostic service. It cannot confirm specimen integrity or replace a '
    'doctor, pathologist, or qualified laboratory professional. Seek urgent medical '
    'care for severe symptoms or a result your laboratory marks as critical.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Gemini client and helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Read the API key from Streamlit secrets first, then environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY")


def get_client() -> genai.Client | None:
    key = get_api_key()
    if not key:
        return None
    return genai.Client(api_key=key)


def model_name() -> str:
    """Allow model changes without editing source code."""
    try:
        configured = st.secrets.get("GEMINI_MODEL", "")
    except Exception:
        configured = ""
    return configured or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON, tolerating a fenced response if one is returned."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ValueError("The AI response was not valid JSON.")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("The AI response did not contain a JSON object.")
    return data


def call_gemini(contents: list[Any], system_instruction: str) -> str:
    """Make a bounded Gemini request and return text."""
    client = get_client()
    if client is None:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY in Streamlit secrets."
        )
    response = client.models.generate_content(
        model=model_name(),
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            max_output_tokens=6000,
        ),
    )
    result = getattr(response, "text", None)
    if not result:
        raise RuntimeError(
            "Gemini returned no readable text. Try again or upload a clearer report."
        )
    return result


def safe_text(value: Any, default: str = "Not stated") -> str:
    if value is None or str(value).strip() in ("", "None", "null"):
        return default
    return str(value).strip()


def normalize_report_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize expected keys without making up missing values."""
    tests = data.get("tests", [])
    if not isinstance(tests, list):
        tests = []
    normalized = []
    for item in tests:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "test_name": safe_text(item.get("test_name"), "Unclear test name"),
            "result": safe_text(item.get("result"), "Unclear"),
            "numeric_value": item.get("numeric_value"),
            "unit": safe_text(item.get("unit"), ""),
            "reference_range": safe_text(item.get("reference_range"), ""),
            "flag_from_report": safe_text(item.get("flag_from_report"), ""),
            "plain_explanation_en": safe_text(item.get("plain_explanation_en"), ""),
            "plain_explanation_ur": safe_text(item.get("plain_explanation_ur"), ""),
            "confidence": safe_text(item.get("confidence"), "Needs review"),
        })
    return {
        "readability": safe_text(data.get("readability"), "uncertain"),
        "report_type": safe_text(data.get("report_type")),
        "patient_name": safe_text(data.get("patient_name")),
        "patient_age": safe_text(data.get("patient_age")),
        "patient_sex": safe_text(data.get("patient_sex")),
        "collection_date_time": safe_text(data.get("collection_date_time")),
        "laboratory_comments": safe_text(data.get("laboratory_comments")),
        "overall_summary_en": safe_text(data.get("overall_summary_en")),
        "overall_summary_ur": safe_text(data.get("overall_summary_ur")),
        "tests": normalized,
        "uncertainties": data.get("uncertainties", []),
    }


REPORT_SYSTEM = """
You are a cautious medical laboratory report-reading assistant. Extract only
what is visibly present in the supplied report. Do not diagnose disease, infer
a patient's condition, or invent values, units, dates, flags, or reference
intervals. Preserve decimal points, signs, units, and reference ranges exactly.
If a field is unreadable or absent, use an empty string or null and explain it
in uncertainties. A reference range must come from the report itself; do not
supply a remembered or generic range. Do not infer sex or age. Use simple,
non-alarming English and Urdu explanations. Explain that an out-of-range result
is not a diagnosis and may have multiple causes. If image quality is poor or
the document does not appear to be a lab report, set readability to "poor".
For numeric_value, provide a number only when clearly legible and numeric;
otherwise null. Return valid JSON only, with this structure:
{
 "readability":"good|partial|poor",
 "report_type":"",
 "patient_name":"",
 "patient_age":"",
 "patient_sex":"",
 "collection_date_time":"",
 "laboratory_comments":"",
 "overall_summary_en":"",
 "overall_summary_ur":"",
 "tests":[
  {"test_name":"","result":"","numeric_value":null,"unit":"",
   "reference_range":"","flag_from_report":"",
   "plain_explanation_en":"","plain_explanation_ur":"",
   "confidence":"high|medium|low"}
 ],
 "uncertainties":[]
}
"""

PREANALYTICAL_SYSTEM = """
You are a cautious clinical laboratory pre-analytical safety education assistant.
Assess only evidence explicitly provided in the report or user answers. Do not
claim to detect hemolysis, clotting, contamination, tube type, volume, labeling,
or processing delay from ordinary chemistry/CBC result values alone. Distinguish
"reported in document", "user-reported", and "not verifiable from supplied
information". Never reject a specimen or make a final suitability decision.
Use alert levels High Priority, Warning, Informational. High Priority is only
for a clearly documented serious identification/safety issue or an explicit
critical laboratory comment requiring immediate laboratory contact; otherwise
use Warning or Informational. Explain possible effect cautiously and give
corrective action / verification guidance. Return valid JSON only:
{
 "overall_assessment":"",
 "alerts":[
  {"priority":"High Priority|Warning|Informational",
   "issue":"","evidence_source":"Report|User-provided|Not verifiable",
   "evidence":"","possible_effect":"",
   "recommended_action":""}
 ],
 "recollection_guidance":"",
 "limitations":[]
}
"""


def analyze_report(uploaded_file, language: str) -> dict[str, Any]:
    """Send uploaded image/PDF bytes directly to Gemini; no server-side saving."""
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type
    prompt = (
        "Read this uploaded medical laboratory report. Preferred explanation "
        f"language: {language}. Extract the visible details and return the JSON "
        "structure specified in your instructions. Be conservative with OCR."
    )
    part = types.Part.from_bytes(data=raw, mime_type=mime)
    raw_response = call_gemini([prompt, part], REPORT_SYSTEM)
    return normalize_report_data(extract_json(raw_response))


def compare_to_report_range(test: dict[str, Any]) -> str:
    """
    Compare a numeric result only with a simple numeric interval printed on
    the report. Complex ranges (age/sex-specific, qualitative, '<', '>') are
    left unclassified rather than guessed.
    """
    value = test.get("numeric_value")
    reference = test.get("reference_range", "")
    if not isinstance(value, (int, float)) or not reference:
        return "Not assessed"
    match = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(-?\d+(?:\.\d+)?)\s*",
        str(reference),
        flags=re.I,
    )
    if not match:
        return "Not assessed"
    low, high = float(match.group(1)), float(match.group(2))
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Normal"


def display_report(data: dict[str, Any], language: str) -> None:
    tests = data["tests"]
    if data["readability"].lower() == "poor" or not tests:
        st.error(
            "The report could not be read reliably. Please upload a sharper, "
            "well-lit image or a clear PDF showing the full page."
        )
        if data.get("uncertainties"):
            st.write("Unclear items:", data["uncertainties"])
        return

    st.subheader("Report overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tests extracted", len(tests))
    c2.metric("Report readability", data["readability"].title())
    abnormal_count = sum(
        compare_to_report_range(t) in ("Low", "High") for t in tests
    )
    c3.metric("Outside printed range", abnormal_count)

    summary_key = "overall_summary_ur" if language == "Urdu" else "overall_summary_en"
    st.info(data.get(summary_key, "No summary was available."))

    rows = []
    for test in tests:
        computed = compare_to_report_range(test)
        rows.append({
            "Test": test["test_name"],
            "Result": test["result"],
            "Unit": test["unit"],
            "Reference range (from report)": test["reference_range"] or "Not provided",
            "Comparison": computed,
            "Report flag": test["flag_from_report"],
            "Reading confidence": test["confidence"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Comparison": st.column_config.TextColumn(
                "Comparison", help="Calculated only for a simple numeric range printed on this report."
            )
        },
    )

    assessed = [r for r in rows if r["Comparison"] in ("Low", "High", "Normal")]
    if assessed:
        counts = pd.Series([r["Comparison"] for r in assessed]).value_counts()
        chart_df = pd.DataFrame({"Status": counts.index, "Number of tests": counts.values})
        st.bar_chart(
            chart_df.set_index("Status"),
            y="Number of tests",
            use_container_width=True,
        )

    st.subheader("Test-by-test explanation")
    for test in tests:
        status = compare_to_report_range(test)
        title = f"{test['test_name']} — {status}"
        with st.expander(title):
            st.write(f"**Result:** {test['result']} {test['unit']}")
            st.write(f"**Reference range on report:** {test['reference_range'] or 'Not provided'}")
            st.write(f"**Report flag:** {test['flag_from_report'] or 'No flag visible'}")
            explanation = (
                test["plain_explanation_ur"] if language == "Urdu"
                else test["plain_explanation_en"]
            )
            st.write(explanation or "No reliable explanation could be generated.")
            if status in ("Low", "High"):
                st.warning(
                    "This result is outside the interval printed on this report. "
                    "It is not a diagnosis; discuss it with your healthcare professional."
                )
            elif status == "Not assessed":
                st.caption(
                    "No automatic comparison was made: a clear numeric interval "
                    "printed on the report was not available."
                )
            if test["confidence"].lower() != "high":
                st.caption("Please verify this extracted item against the original report.")

    if data.get("laboratory_comments") != "Not stated":
        st.subheader("Laboratory comments")
        st.write(data["laboratory_comments"])
    if data.get("uncertainties"):
        st.warning("Items requiring manual verification")
        for item in data["uncertainties"]:
            st.write(f"• {item}")


# ---------------------------------------------------------------------------
# Sidebar: settings and patient context
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    language = st.radio("Explanation language", ["English", "Urdu"], horizontal=True)
    st.caption(f"Gemini model: `{model_name()}`")
    if get_api_key():
        st.success("Gemini API key detected")
    else:
        st.warning("Add your Gemini API key to enable AI features.")
    st.divider()
    st.subheader("Optional patient context")
    patient_age = st.number_input("Age (years)", min_value=0, max_value=120, value=0)
    patient_sex = st.selectbox("Sex (if relevant to the test reference interval)", [
        "Not provided", "Female", "Male", "Other / prefer not to say"
    ])
    st.caption(
        "Context is not used to invent reference ranges. Use the interval "
        "printed by the performing laboratory whenever possible."
    )


# ---------------------------------------------------------------------------
# Main application tabs
# ---------------------------------------------------------------------------

tab_upload, tab_analysis, tab_pre, tab_check, tab_chat = st.tabs([
    "📤 Upload report",
    "📊 Report analysis",
    "🛡️ Pre-analytical review",
    "☑️ Collection checklist",
    "💬 AI medical assistant",
])

with tab_upload:
    st.subheader("Upload a medical laboratory report")
    st.write("Supported formats: JPG, JPEG, PNG, PDF. Maximum file size: 15 MB.")
    uploaded = st.file_uploader(
        "Choose report image or PDF",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=False,
        help="Upload a clear, complete page. Avoid glare, shadows, and cropped values.",
        key="report_upload",
    )
    if uploaded:
        if uploaded.size > 15 * 1024 * 1024:
            st.error("File is larger than 15 MB. Please upload a smaller file.")
            uploaded = None
        else:
            if uploaded.type.startswith("image/"):
                st.image(uploaded, caption="Uploaded report preview", use_container_width=True)
            else:
                st.success(f"PDF selected: {uploaded.name}")
            st.caption(
                "Privacy: this app does not write the uploaded report to its own "
                "disk or database. Its bytes are sent to Google Gemini for analysis "
                "when you press the button. Review Google's applicable data terms."
            )
            if st.button("Analyse report with Gemini", type="primary", use_container_width=True):
                with st.spinner("Reading report carefully…"):
                    try:
                        result = analyze_report(uploaded, language)
                        st.session_state["report_analysis"] = result
                        st.session_state["report_filename"] = uploaded.name
                        st.success("Analysis received. Verify every extracted value against the original.")
                    except Exception as exc:
                        st.error(f"Could not analyse this report: {exc}")
                        st.info("Check your API key, model access, internet connection, and file clarity.")

with tab_analysis:
    st.subheader("Medical report analysis")
    data = st.session_state.get("report_analysis")
    if data:
        st.caption(f"Source file: {st.session_state.get('report_filename', 'Uploaded report')}")
        display_report(data, language)
    else:
        st.info("Upload a report and select **Analyse report with Gemini** to view the results here.")

with tab_pre:
    st.subheader("Pre-analytical error prevention")
    st.write(
        "This review combines report comments with information you provide. "
        "It cannot visually confirm hemolysis, clotting, tube choice, sample volume, "
        "or transport conditions unless those facts are explicitly documented."
    )
    report_data = st.session_state.get("report_analysis", {})
    lab_comments = report_data.get("laboratory_comments", "Not stated")
    st.text_area("Laboratory comments extracted from report", value=lab_comments,
                 disabled=True, key="lab_comments_display")

    with st.form("preanalytical_form"):
        st.markdown("**Patient and sample details**")
        p_name = st.text_input("Patient name or identifier (optional)")
        report_name = st.text_input("Name/identifier shown on report (optional)")
        collection_dt = st.text_input("Collection date and time (if known)")
        sample_type = st.selectbox("Specimen type", [
            "Not sure", "Blood", "Urine", "Stool", "Swab", "Other"
        ])
        st.markdown("**Sample condition and handling — choose only what you know**")
        hemolysis = st.selectbox("Was hemolysis reported or observed?", [
            "Not known / not assessed", "Yes", "No"
        ])
        clot = st.selectbox("Was clotting reported or observed (for a test requiring anticoagulated blood)?", [
            "Not known / not assessed", "Yes", "No"
        ])
        volume = st.selectbox("Was sample volume reported as insufficient?", [
            "Not known / not assessed", "Yes", "No"
        ])
        tube = st.selectbox("Was the collection tube confirmed correct for the requested test?", [
            "Not known", "Yes", "No", "Not applicable"
        ])
        label = st.selectbox("Was the sample labelled at the patient's side with correct identifiers?", [
            "Not known", "Yes", "No"
        ])
        delay = st.selectbox("Was there a known delay before processing?", [
            "Not known", "Yes", "No"
        ])
        storage = st.selectbox("Was storage/transport outside the laboratory's instructions?", [
            "Not known", "Yes", "No"
        ])
        fasting = st.selectbox("Were fasting instructions followed when required?", [
            "Not applicable / not known", "Yes", "No"
        ])
        other_info = st.text_area(
            "Other known collection or transport details",
            placeholder="Example: sample collected after a difficult draw; lab called to request recollection."
        )
        submitted_pre = st.form_submit_button("Review potential pre-analytical concerns", type="primary")

    if submitted_pre:
        if not get_api_key():
            st.error("Gemini API key is required for this review.")
        else:
            evidence = {
                "patient_identifier_entered_by_user": p_name or "Not provided",
                "identifier_on_report_entered_by_user": report_name or "Not provided",
                "collection_date_time": collection_dt or "Not provided",
                "specimen_type": sample_type,
                "hemolysis": hemolysis,
                "clotting": clot,
                "insufficient_volume": volume,
                "correct_tube_confirmed": tube,
                "labeling_at_patient_side": label,
                "processing_delay": delay,
                "storage_transport_issue": storage,
                "fasting": fasting,
                "additional_user_information": other_info or "None",
                "laboratory_comments_from_report": lab_comments,
            }
            prompt = (
                "Assess this information for possible pre-analytical concerns. "
                "Do not assume any unknown item is an error. If both identifiers "
                "are provided and differ, flag for urgent identity verification; "
                "do not claim a mismatch when either is absent. User answers: "
                + json.dumps(evidence, ensure_ascii=False)
            )
            with st.spinner("Reviewing evidence and limitations…"):
                try:
                    pre_result = extract_json(call_gemini([prompt], PREANALYTICAL_SYSTEM))
                    st.session_state["pre_analysis"] = pre_result
                except Exception as exc:
                    st.error(f"Review failed: {exc}")

    pre_result = st.session_state.get("pre_analysis")
    if pre_result:
        st.markdown("### Assessment")
        st.write(pre_result.get("overall_assessment", ""))
        for alert in pre_result.get("alerts", []):
            priority = alert.get("priority", "Informational")
            message = (
                f"**{priority}: {alert.get('issue', 'Potential concern')}**\n\n"
                f"Evidence source: {alert.get('evidence_source', 'Not stated')}\n\n"
                f"Evidence: {alert.get('evidence', '')}\n\n"
                f"Possible effect: {alert.get('possible_effect', '')}\n\n"
                f"Recommended action: {alert.get('recommended_action', '')}"
            )
            if priority == "High Priority":
                st.error(message)
            elif priority == "Warning":
                st.warning(message)
            else:
                st.info(message)
        st.markdown("### Recollection / verification guidance")
        st.write(pre_result.get("recollection_guidance", ""))
        if pre_result.get("limitations"):
            st.markdown("### What cannot be verified")
            for item in pre_result["limitations"]:
                st.write(f"• {item}")
        st.caption(
            "These are decision-support alerts, not specimen acceptance/rejection "
            "decisions. Contact the performing laboratory; only qualified staff "
            "can determine specimen suitability and recollection requirements."
        )

with tab_check:
    st.subheader("Sample collection checklist")
    st.write(
        "Use this as a discussion aid. Requirements vary by test, collection tube, "
        "laboratory SOP, and local policy. Follow the instructions from the ordering "
        "clinician and performing laboratory."
    )
    checklist_groups = {
        "Before collection": [
            "Patient identity checked using the laboratory's required identifiers",
            "Test request and specimen type confirmed",
            "Fasting requirement checked for the specific test",
            "Patient preparation and relevant medicines/instructions reviewed",
            "Collection date and time recorded where required",
        ],
        "During collection": [
            "Correct collection container/tube selected for the requested test",
            "Order of draw followed when multiple blood tubes are collected",
            "Tube filled to the required volume / fill line",
            "Specimen labelled immediately at the patient's side",
            "Labels match the request form and patient identifiers",
        ],
        "After collection": [
            "Specimen mixed gently when required by tube instructions",
            "Specimen checked for visible leakage or other obvious issue",
            "Required transport temperature and protection followed",
            "Specimen transported and processed within the test-specific time limit",
            "Any collection difficulty or incident communicated to the laboratory",
        ],
    }
    checked_items = []
    for group, items in checklist_groups.items():
        st.markdown(f"**{group}**")
        cols = st.columns(2)
        for index, item in enumerate(items):
            with cols[index % 2]:
                if st.checkbox(item, key=f"check_{group}_{index}"):
                    checked_items.append(item)
    total_items = sum(len(items) for items in checklist_groups.values())
    st.progress(len(checked_items) / total_items)
    st.caption(f"{len(checked_items)} of {total_items} checklist items marked complete.")
    st.info(
        "Order of draw and tube selection are test- and laboratory-dependent. "
        "Do not use a generic tube-colour chart as a substitute for the local SOP."
    )

with tab_chat:
    st.subheader("AI medical laboratory assistant")
    st.write("Ask about a test, a result in your uploaded report, fasting, or sample collection.")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Type your question in English or Urdu…")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Preparing an educational answer…"):
                try:
                    context = ""
                    current = st.session_state.get("report_analysis")
                    if current:
                        context = (
                            "\nThe user has an uploaded report. Use only this extracted "
                            "context, and remind them to verify against the original:\n"
                            + json.dumps(current, ensure_ascii=False)[:12000]
                        )
                    system = (
                        "You are an educational medical laboratory assistant. Reply "
                        "in the language used by the user (simple English or Urdu). "
                        "Explain general lab concepts clearly. Do not diagnose, prescribe, "
                        "or claim a pre-analytical error without evidence. Do not invent "
                        "reference ranges. If a result is discussed, explain that ranges "
                        "depend on the lab, method, age, sex, pregnancy status, and context. "
                        "For critical flags, severe symptoms, or urgent concerns, advise "
                        "prompt contact with a clinician or laboratory. Never replace "
                        "professional advice. Keep the answer concise."
                    )
                    answer = call_gemini([system + context, question], system)
                    st.markdown(answer)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as exc:
                    error_msg = f"I couldn't answer just now: {exc}"
                    st.error(error_msg)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": error_msg}
                    )

st.divider()
st.markdown(
    """
    <p class="muted">
    LabLens is a prototype for education and workflow support. Laboratory reference
    intervals and specimen acceptance criteria must be validated locally before
    clinical use. The app does not retain uploaded report files in its own storage,
    but information sent to an AI provider is subject to that provider's terms.
    </p>
    """,
    unsafe_allow_html=True,
)
