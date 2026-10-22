import streamlit as st
import requests
import pandas as pd

# Backend API Base URL
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="FinSight AI — Financial Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SIDEBAR: SYSTEM HEALTH & UPLOAD -----------------
with st.sidebar:
    st.title("📈 FinSight AI")
    st.caption("Autonomous Financial Research & Valuation")
    st.divider()

    # Health Probe
    st.subheader("System Status")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            data = health_resp.json()
            st.success(f"🟢 API Online ({data['environment'].capitalize()} v{data['version']})")
        else:
            st.error("🔴 API Error")
    except Exception:
        st.error("🔴 Backend Offline\nRun: `python -m src.api`")

    st.divider()

    # Document Uploader
    st.subheader("Ingest SEC Filing")
    uploaded_file = st.file_uploader("Upload 10-K / 10-Q (PDF)", type=["pdf"])
    if uploaded_file and st.button("Index Document in ChromaDB", use_container_width=True):
        with st.spinner("Extracting text, cleaning tables, and generating 3,072-dim embeddings..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                upload_resp = requests.post(f"{API_BASE_URL}/api/upload", files=files, timeout=60)
                if upload_resp.status_code == 201:
                    res = upload_resp.json()
                    st.success(f" Indexed {res['filename']} ({res['total_pages']} pages, {res['total_chunks']} chunks)")
                    st.rerun()
                else:
                    st.error(f"Failed to ingest: {upload_resp.text}")
            except Exception as e:
                st.error(f"Upload error: {e}")

    st.divider()

    # Document Catalog
    st.subheader("Active Document Catalog")
    try:
        docs_resp = requests.get(f"{API_BASE_URL}/api/documents", timeout=3)
        if docs_resp.status_code == 200 and docs_resp.json():
            for doc in docs_resp.json():
                st.markdown(f"📄 **{doc['filename']}**  \n`{doc['total_pages']} pages | {doc['total_chunks']} chunks`")
        else:
            st.caption("No documents in database catalog yet.")
    except Exception:
        st.caption("Could not load catalog.")

# ----------------- MAIN CONTENT: TABS -----------------
st.title("Financial Intelligence & Quantitative Research")

tab_rag, tab_agent, tab_audit = st.tabs([
    "🔍 Grounded RAG Analyst",
    "🤖 Autonomous Quantitative Agent",
    "📊 Compliance Audit & Latency"
])

# ----------------- TAB 1: GROUNDED RAG -----------------
with tab_rag:
    st.subheader("Factual Document Question-Answering")
    st.caption("Grounded retrieval directly from ChromaDB with verifiable page citations.")

    col1, col2 = st.columns([4, 1])
    with col1:
        rag_query = st.text_input(
            "Enter your financial question:",
            placeholder="e.g., What were Apple's Products and Services sales for the three months ended June 27, 2026?",
            key="rag_input"
        )
    with col2:
        top_k = st.slider("Retrieval Depth (top_k)", min_value=1, max_value=5, value=2)

    if st.button("Run Grounded RAG", type="primary"):
        if not rag_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving document chunks and generating grounded answer..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/api/rag",
                        json={"question": rag_query, "top_k": top_k},
                        timeout=30
                    )
                    if resp.status_code == 200:
                        rag_result = resp.json()
                        st.markdown("### Analyst Response")
                        st.info(rag_result["answer"])

                        st.markdown("#### Retrieved Source Citations")
                        for idx, src in enumerate(rag_result.get("sources", []), 1):
                            st.markdown(f"- **Citation #{idx}**: Page `{src['page']}` | Cosine Similarity: `{src['similarity']:.4f}`")
                    else:
                        st.error(f"Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"API Connection Error: {e}")

# ----------------- TAB 2: AUTONOMOUS AGENT -----------------
with tab_agent:
    st.subheader("Multi-Step ReAct Quantitative Research")
    st.caption("Autonomous Agent dynamically searches filings, extracts figures, and executes verified math tools.")

    agent_prompt = st.text_area(
        "Enter your research and calculation prompt:",
        value="For the three months ended June 27, 2026, find Apple's Products revenue and Services revenue, and calculate what percentage Services is of Products.",
        height=100,
        key="agent_input"
    )

    if st.button("Execute Agent Loop", type="primary"):
        if not agent_prompt.strip():
            st.warning("Please enter a research prompt.")
        else:
            with st.spinner("Agent reasoning: searching document, executing math tools, synthesizing report..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/api/agent",
                        json={"prompt": agent_prompt},
                        timeout=45
                    )
                    if resp.status_code == 200:
                        agent_res = resp.json()
                        st.markdown("### Agent Synthesis")
                        st.success(agent_res["answer"])
                    else:
                        st.error(f"Agent Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"API Connection Error: {e}")

# ----------------- TAB 3: AUDIT TRAIL & LATENCY -----------------
with tab_audit:
    st.subheader("SEC Rule 17a-4 Compliance Audit Trail")
    st.caption("Live relational audit logs from SQLite / PostgreSQL with execution latency tracking.")

    if st.button("Refresh Audit Logs"):
        st.rerun()

    try:
        audit_resp = requests.get(f"{API_BASE_URL}/api/audit-logs?limit=50", timeout=5)
        if audit_resp.status_code == 200 and audit_resp.json():
            logs_data = audit_resp.json()
            df = pd.DataFrame(logs_data)
            df["created_at"] = pd.to_datetime(df["created_at"])

            # Metric summary
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Queries Logged", len(df))
            col_m2.metric("Average Latency", f"{df['execution_time_ms'].mean():.1f} ms")
            col_m3.metric("Max Latency", f"{df['execution_time_ms'].max():.1f} ms")

            st.markdown("#### Query Latency (ms)")
            st.bar_chart(df.set_index("id")["execution_time_ms"])

            st.markdown("#### Recent Audit Records")
            st.dataframe(
                df[["id", "endpoint", "query_text", "execution_time_ms", "sources_count", "created_at"]],
                use_container_width=True
            )
        else:
            st.info("No audit logs recorded yet. Execute queries in Tab 1 or Tab 2 to generate logs.")
    except Exception as e:
        st.error(f"Could not load audit logs: {e}")