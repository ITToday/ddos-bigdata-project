# ============================================================
# DDoS Big Data Analytics and Recommendation System
# Streamlit Application
# Student: Benjamine (Batch 18, MSc AI, University of Moratuwa)
# Module: IT5612 Big Data Analytics Mini Project
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="DDoS Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h2 { font-size: 20px; font-weight: 600; color: #1a1a2e; margin-bottom: 4px; }
    h3 { font-size: 15px; font-weight: 600; color: #1a1a2e; }
    h4 { font-size: 13px; font-weight: 600; color: #374151; }
    .metric-card {
        background: #f8f9fa; border: 1px solid #e5e7eb;
        border-radius: 6px; padding: 16px 12px; text-align: center;
    }
    .metric-value { font-size: 26px; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 12px; color: #6b7280; margin-top: 4px; }
    .metric-sub   { font-size: 11px; color: #9ca3af; margin-top: 2px; }
    .finding-box {
        border-left: 3px solid #1a1a2e; padding: 8px 14px;
        margin-bottom: 8px; background: #f8f9fa;
        border-radius: 0 4px 4px 0;
    }
    .finding-box p { margin: 0; font-size: 13px; color: #374151; line-height: 1.5; }
    .llm-card {
        background: #f0f4ff; border: 1px solid #c7d2fe;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;
    }
    .llm-rank { font-size: 13px; font-weight: 700; color: #1a1a2e; }
    .llm-just { font-size: 12px; color: #4b5563; margin-top: 4px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================
BASE                = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DISTRIBUTION_PATH   = os.path.join(BASE, 'distribution.csv')
CENTROIDS_PATH      = os.path.join(BASE, 'attack_type_centroids.parquet')
MATRIX_PATH         = os.path.join(BASE, 'interaction_matrix.csv')
DISCRIMINATION_PATH = os.path.join(BASE, 'discrimination.json')
ALS_PATH            = os.path.join(BASE, 'als_results.json')
EVAL_PATH           = os.path.join(BASE, 'evaluation_results.json')

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_distribution():
    return pd.read_csv(DISTRIBUTION_PATH)

@st.cache_data
def load_centroids():
    return pd.read_parquet(CENTROIDS_PATH)

@st.cache_data
def load_interaction_matrix():
    return pd.read_csv(MATRIX_PATH, index_col=0)

@st.cache_data
def load_discrimination():
    with open(DISCRIMINATION_PATH) as f:
        return pd.DataFrame(json.load(f))

@st.cache_data
def load_als():
    if os.path.exists(ALS_PATH):
        with open(ALS_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_eval():
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data
def compute_cf_similarity(matrix_pd):
    v = matrix_pd.values.astype(float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    norm = v / n
    sim  = np.dot(norm, norm.T)
    return pd.DataFrame(sim, index=matrix_pd.index, columns=matrix_pd.index)

@st.cache_data
def compute_content_similarity(centroids_pd):
    cols   = [c for c in centroids_pd.columns if c.startswith("avg_")]
    mat    = centroids_pd[cols].fillna(0)
    scaled = StandardScaler().fit_transform(mat.values)
    n      = np.linalg.norm(scaled, axis=1, keepdims=True)
    n[n == 0] = 1
    norm   = scaled / n
    sim    = np.dot(norm, norm.T)
    return pd.DataFrame(sim, index=mat.index, columns=mat.index)

try:
    dist_df   = load_distribution()
    centroids = load_centroids().set_index("activity")
    matrix_pd = load_interaction_matrix()
    disc_df   = load_discrimination()
    als_res   = load_als()
    eval_res  = load_eval()
    cf_sim    = compute_cf_similarity(matrix_pd)
    cb_sim    = compute_content_similarity(centroids)
    data_ok   = True
except Exception as e:
    data_ok  = False
    data_err = str(e)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## DDoS Analytics Platform")
st.sidebar.markdown(
    "MSc AI — Big Data Analytics  \n"
    "University of Moratuwa  \n"
    "Student: Benjamine (Batch 18)"
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select section",
    [
        "Overview",
        "Q1 — Attack Distribution",
        "Q2 — Feature Signatures",
        "Q3 — Timing Patterns",
        "Q4 — Flag Signatures",
        "Q5 — Discrimination",
        "Recommendation System"
    ]
)

st.sidebar.markdown("---")

if data_ok:
    n_atk = len(dist_df[dist_df['label'] == 'Attack'])
    n_ben = len(dist_df[dist_df['label'] == 'Benign'])
    st.sidebar.markdown("**Data status**")
    st.sidebar.markdown(
        f"- {n_atk} DDoS attack types  \n"
        f"- {n_ben} benign traffic categories  \n"
        f"- {len(centroids)} feature centroid vectors  \n"
        f"- {int(dist_df['flow_count'].sum()):,} total flows"
    )
else:
    st.sidebar.error(
        f"Output files not found.  \n"
        f"Run Part A and Part B notebooks first.  \n\n"
        f"Error: {data_err}"
    )
    st.stop()

# ============================================================
# HELPERS
# ============================================================
def get_hybrid(query, K=5, alpha=0.6):
    if query not in cf_sim.index or query not in cb_sim.index:
        return []
    cf = cf_sim[query].drop(query, errors='ignore')
    cb = cb_sim[query].drop(query, errors='ignore')
    common = cf.index.intersection(cb.index)
    cf_n = (cf[common] - cf[common].min()) / (cf[common].max() - cf[common].min() + 1e-9)
    cb_n = (cb[common] - cb[common].min()) / (cb[common].max() - cb[common].min() + 1e-9)
    h = (alpha * cf_n + (1 - alpha) * cb_n).sort_values(ascending=False).head(K)
    return [{"attack_type": atk, "hybrid_score": round(float(s), 4),
             "cf_score": round(float(cf_n[atk]), 4),
             "content_score": round(float(cb_n[atk]), 4)}
            for atk, s in h.items()]

def build_profile_text(query, centroids_pd):
    if query not in centroids_pd.index:
        return f"Attack type: {query}"
    row    = centroids_pd.loc[query]
    labels = {
        "avg_bytes_rate": "Avg bytes rate",
        "avg_packets_rate": "Avg packets/sec",
        "avg_duration": "Avg flow duration (s)",
        "avg_payload_bytes_mean": "Avg payload size (bytes)",
        "avg_syn_flag_counts": "Avg SYN flag count",
        "avg_ack_flag_counts": "Avg ACK flag count",
        "avg_down_up_rate": "Down/up traffic ratio",
        "avg_fwd_packets_rate": "Forward packets/sec",
        "avg_bwd_packets_rate": "Backward packets/sec"
    }
    lines = [f"Observed attack type: {query}"]
    for col, label in labels.items():
        if col in row.index:
            lines.append(f"  {label}: {row[col]:.4f}")
    return "\n".join(lines)

def call_llm(query, candidates, centroids_pd, api_key):
    from openai import OpenAI
    client   = OpenAI(api_key=api_key)
    profile  = build_profile_text(query, centroids_pd)
    cand_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    prompt   = f"""You are a cybersecurity expert specialising in DDoS attack analysis.

OBSERVED ATTACK PROFILE:
{profile}

HYBRID RECOMMENDATION CANDIDATES:
{cand_str}

TASK:
Re-rank these candidates from most to least relevant for a SOC analyst
to prepare defences against, based on the observed traffic features.
Consider packet rate, flow duration, flag patterns, and bytes rate.

Return ONLY valid JSON, no markdown, no extra text:
{{
  "ranked_recommendations": [
    {{"rank": 1, "attack_type": "...", "justification": "one sentence grounded in traffic features"}},
    {{"rank": 2, "attack_type": "...", "justification": "one sentence"}},
    {{"rank": 3, "attack_type": "...", "justification": "one sentence"}},
    {{"rank": 4, "attack_type": "...", "justification": "one sentence"}},
    {{"rank": 5, "attack_type": "...", "justification": "one sentence"}}
  ]
}}"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=700
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)["ranked_recommendations"]
    except:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)["ranked_recommendations"]

# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":

    st.markdown("## DDoS Network Traffic Analytics and Recommendation System")
    st.markdown(
        "This system analyses 540,494 DDoS network flow records using Apache Spark "
        "and recommends which attack types a SOC analyst should prepare defences for, "
        "given an observed attack on a network service."
    )
    st.markdown("---")

    total = int(dist_df['flow_count'].sum())
    atk   = int(dist_df[dist_df['label'] == 'Attack']['flow_count'].sum())
    ben   = int(dist_df[dist_df['label'] == 'Benign']['flow_count'].sum())
    sus   = int(dist_df[dist_df['label'] == 'Suspicious']['flow_count'].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total flows analysed</div>
            <div class="metric-sub">Apache Spark distributed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#dc2626">{atk:,}</div>
            <div class="metric-label">DDoS attack flows</div>
            <div class="metric-sub">{atk/total*100:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#1d4ed8">{ben:,}</div>
            <div class="metric-label">Benign flows</div>
            <div class="metric-sub">{ben/total*100:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#d97706">{sus:,}</div>
            <div class="metric-label">Suspicious flows</div>
            <div class="metric-sub">{sus/total*100:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Part A — Big Data Analytics Findings")
        for title, desc in [
            ("Two-level classification structure",
             "The dataset uses label (3 coarse categories) and activity "
             "(26 specific traffic scenarios). All analysis groups by activity "
             "to preserve variation between individual attack types."),
            ("Severe class imbalance",
             "Attack-TCP-BYPass-V1 accounts for 78.6% of all attack flows. "
             "The remaining 16 types have 625 to 3,147 flows — rare but "
             "operationally dangerous as they evade threshold-based detection."),
            ("Distinct per-attack traffic signatures",
             "Each of the 17 attack types has a measurable and distinct "
             "profile across bytes rate, packet rate, duration, and TCP flags."),
            ("BYPass-V1 operates differently from all others",
             "Lowest packet rate (61.55/sec) but highest flow count. Exhausts "
             "server connection state through volume of attempts, not bandwidth."),
            ("Top discriminating features identified",
             "fwd_init_win_bytes (ratio 4.12) and packets_IAT_mean (ratio 1.75) "
             "most clearly separate attack from benign — validating the feature "
             "vectors used in the recommendation system.")
        ]:
            st.markdown(f"""<div class="finding-box">
                <p><strong>{title}</strong><br>{desc}</p>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("#### Part B — Recommendation System")
        for title, desc in [
            ("Problem statement",
             "A SOC analyst observing a DDoS attack needs to know which other "
             "attack types to prepare defences for. This system answers that "
             "question given an observed attack type and targeted network service."),
            ("Users, items, and ratings",
             "Users = network service profiles grouped by destination port. "
             "Items = 17 DDoS attack types. "
             "Ratings = flow count as implicit feedback — the number of flows "
             "of each attack type observed targeting each service."),
            ("Four recommendation layers",
             "Layer 1: Collaborative filtering — attack types that co-occur "
             "across similar service profiles.  "
             "Layer 2: ALS matrix factorization — latent factor learning "
             "via pyspark.ml.  "
             "Layer 3: Content-based — attack types with similar feature profiles.  "
             "Layer 4: Hybrid weighted combination.  "
             "Layer 5: LLM reflection — GPT re-ranks using traffic profile reasoning."),
            ("Evaluation results",
             f"Precision@5 = {eval_res.get('precision_at_5', 1.0):.4f} — all "
             f"top-5 recommendations are genuinely observed attack types.  "
             f"Recall@5 = {eval_res.get('recall_at_5', 0.294):.4f} — top-5 "
             f"covers 5 of 17 possible attack types.  "
             f"ALS RMSE = {eval_res.get('rmse', 859):.2f} — expected for "
             f"implicit feedback on high-variance count data.")
        ]:
            st.markdown(f"""<div class="finding-box">
                <p><strong>{title}</strong><br>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    left2, right2 = st.columns(2)

    with left2:
        st.markdown("#### Part A — Analytical Pipeline")
        st.dataframe(pd.DataFrame([
            ["Q1", "Attack type distribution",
             "GROUP BY + SUM() OVER() window function"],
            ["Q2", "Feature signatures per attack type",
             "groupBy().agg() programmatic expressions"],
            ["Q3", "Timing patterns and intensity ranking",
             "RANK() OVER() window function"],
            ["Q4", "TCP flag combinations per attack",
             "AVG() grouped by activity, normalised heatmap"],
            ["Q5", "Feature discrimination analysis",
             "Mean ratio Attack mean divided by Benign mean"]
        ], columns=["Section", "Analytical Question", "Spark Technique"]),
            use_container_width=True, hide_index=True)

    with right2:
        st.markdown("#### Dataset Information")
        st.dataframe(pd.DataFrame([
            ["Name", "BCCC-cPacket-Cloud-DDoS-2024"],
            ["Source", "York University / cPacket Networks"],
            ["Published", "March 2024"],
            ["License", "CC-BY-SA-4.0"],
            ["Total flows", f"{total:,}"],
            ["Features per flow", "319"],
            ["Traffic scenarios", "26 (17 attack, 8 benign, 1 suspicious)"],
            ["Format", "Parquet (29.5 MB compressed)"],
            ["Processing engine", "Apache Spark 4.0 / PySpark"]
        ], columns=["Property", "Value"]),
            use_container_width=True, hide_index=True)

# ============================================================
# Q1
# ============================================================
elif page == "Q1 — Attack Distribution":

    st.markdown("## Q1 — Attack Type Distribution")
    st.markdown(
        "What is the distribution of specific DDoS attack types? "
        "Which variants are dominant and which are rare?"
    )
    st.markdown("---")

    chart = os.path.join(BASE, 'q1_attack_distribution.png')
    if os.path.exists(chart):
        st.image(chart, use_container_width=True)
    else:
        st.warning("Chart not found. Run Part A notebook first.")

    st.markdown("---")
    st.markdown("#### Distribution Table")
    table = dist_df.sort_values('flow_count', ascending=False).copy()
    table['pct'] = (table['flow_count'] / table['flow_count'].sum() * 100).round(2)
    table.columns = ['Activity', 'Label', 'Flow Count', 'Pct of Total']
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""<div class="finding-box"><p>
        <strong>Finding:</strong> Attack-TCP-BYPass-V1 dominates at 78.6% of all
        attack flows (134,110 flows). The remaining 16 attack types have between
        625 and 3,147 flows each. In a SOC environment, these minority classes are
        the most operationally dangerous — their rarity means fewer alerts and
        higher probability of evading threshold-based detection rules. This directly
        motivates the hybrid recommendation approach in Part B, where content-based
        similarity complements collaborative filtering to surface rare attack types
        that have weak co-occurrence signals.
    </p></div>""", unsafe_allow_html=True)

# ============================================================
# Q2
# ============================================================
elif page == "Q2 — Feature Signatures":

    st.markdown("## Q2 — Traffic Feature Signatures Per Attack Type")
    st.markdown(
        "What are the measurable traffic characteristics of each attack type? "
        "These centroid vectors become the content representation in Part B."
    )
    st.markdown("---")

    attack_list = sorted(centroids.index.tolist())
    selected    = st.selectbox("Select attack type to inspect:", attack_list)

    if selected:
        content_cols = [c for c in centroids.columns if c.startswith("avg_")]
        row = centroids.loc[selected, content_cols]

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Feature Profile")
            st.dataframe(pd.DataFrame({
                "Feature"   : [c.replace("avg_", "") for c in content_cols],
                "Mean Value": [round(float(v), 6) for v in row.values]
            }), use_container_width=True, hide_index=True)

        with col_r:
            st.markdown("#### All Attack Type Profiles")
            all_profiles = centroids[content_cols].copy()
            all_profiles.columns = [c.replace("avg_", "") for c in content_cols]
            st.dataframe(all_profiles.round(4), use_container_width=True)

    st.markdown("---")
    st.markdown("""<div class="finding-box"><p>
        <strong>Finding:</strong> Nine features were selected to cover each meaningful
        traffic dimension without redundancy — volume, intensity, timing, payload size,
        TCP flag exploitation, asymmetry, and directional packet rates. The average
        value per attack type (centroid) is saved and used directly as the content
        vector in Part B Layer 3. This is the bridge between Part A analysis and
        Part B recommendation.
    </p></div>""", unsafe_allow_html=True)

# ============================================================
# Q3
# ============================================================
elif page == "Q3 — Timing Patterns":

    st.markdown("## Q3 — Flow Timing Pattern Analysis")
    st.markdown(
        "How do packet inter-arrival times and flow durations differ? "
        "Timing patterns reveal distinct operational tiers within the attack taxonomy."
    )
    st.markdown("---")

    chart = os.path.join(BASE, 'q3_timing_patterns.png')
    if os.path.exists(chart):
        st.image(chart, use_container_width=True)
    else:
        st.warning("Chart not found. Run Part A notebook first.")

    st.markdown("---")
    st.markdown("""<div class="finding-box"><p>
        <strong>Finding:</strong> Flag-OSYNP has the highest packet rate at
        12,149 packets/sec. Attack-TCP-BYPass-V1 is ranked last at 61.55
        packets/sec but has the highest flow count (134,110 flows). Each BYPass
        flow lasts 0.0025 seconds with 1.02 packets — the attack exhausts server
        connection state through volume of attempts, not packet volume. The
        RANK() OVER() window function computes this ranking across all 26
        activity types in a single query without collapsing rows.
    </p></div>""", unsafe_allow_html=True)

# ============================================================
# Q4
# ============================================================
elif page == "Q4 — Flag Signatures":

    st.markdown("## Q4 — TCP Flag Signature Analysis")
    st.markdown(
        "Which TCP flag combinations characterise each attack type? "
        "Flag patterns reveal the mechanism and evasion strategy."
    )
    st.markdown("---")

    chart = os.path.join(BASE, 'q4_flag_heatmap.png')
    if os.path.exists(chart):
        st.image(chart, use_container_width=True)
    else:
        st.warning("Chart not found. Run Part A notebook first.")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### TCP Flag Reference")
        st.dataframe(pd.DataFrame([
            ["SYN", "SYN floods — half-open connections exhaust server state"],
            ["ACK", "ACK floods — fake established-session traffic"],
            ["PSH", "Combined with ACK in application-layer floods"],
            ["FIN", "FIN floods — abuse connection teardown"],
            ["RST", "RST attacks — force connection termination"],
            ["URG", "Anomalous presence is itself a detection signal"]
        ], columns=["Flag", "Attack Relevance"]),
            use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### Key Observations")
        for title, desc in [
            ("SYN-dominant attacks",
             "BYPass-V1, Flag-OSYN, Flag-OSYNP, Flag-RST-ACK show high SYN "
             "with suppressed ACK and FIN — connection initiation without completion."),
            ("RST-dominant attacks",
             "Killer-TCP, Killall-v2, Flag-ACK all show high RST counts — "
             "deliberately disrupting established sessions."),
            ("Mixed-flag attacks (evasion)",
             "TCP-Control activates SYN, ACK, PSH, and RST simultaneously — "
             "a deliberate technique to evade single-flag threshold rules."),
            ("Benign comparison",
             "Legitimate traffic shows high ACK (TCP acknowledgements) and "
             "high PSH in FTP (file transfer data segments). Benign-FTP "
             "has avg_ack_flag_counts of 6,179 — completely normal for file transfers.")
        ]:
            st.markdown(f"""<div class="finding-box">
                <p><strong>{title}</strong><br>{desc}</p>
            </div>""", unsafe_allow_html=True)

# ============================================================
# Q5
# ============================================================
elif page == "Q5 — Discrimination":

    st.markdown("## Q5 — Feature Discrimination Analysis")
    st.markdown(
        "Which features most clearly separate attack from benign traffic? "
        "This validates the feature vectors used in the Part B recommendation system."
    )
    st.markdown("---")

    chart = os.path.join(BASE, 'q5_discrimination.png')
    if os.path.exists(chart):
        st.image(chart, use_container_width=True)
    else:
        st.warning("Chart not found. Run Part A notebook first.")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Discrimination Ratios")
        disc_table = disc_df.copy()
        disc_table.columns = ["Feature", "Ratio", "Direction"]
        disc_table["Direction"] = disc_table["Direction"].map({
            "attack": "Attack higher",
            "benign": "Benign higher"
        })
        st.dataframe(
            disc_table.sort_values("Ratio", ascending=False),
            use_container_width=True, hide_index=True
        )

    with col_r:
        st.markdown("#### Interpretation")
        for title, desc in [
            ("fwd_init_win_bytes — ratio 4.12 (Attack higher)",
             "Attack traffic uses abnormally large TCP initial window sizes. "
             "BYPass-V1 manipulates the window field to evade stateful inspection. "
             "Benign traffic uses standard negotiated window sizes."),
            ("packets_IAT_mean — ratio 1.75 (Attack higher)",
             "Attack flows are extremely short (0.0025 sec average) with "
             "measurable gaps at session level, producing higher mean IAT "
             "than sustained benign connections."),
            ("All remaining — Benign higher",
             "Benign traffic includes FTP (bytes_rate 2.5 million) and web "
             "browsing (ack_flag_counts 159) which pull benign class means "
             "above attack means. Attack traffic carries minimal payload."),
            ("Part B connection",
             "These 14 features confirm that the 9-feature centroid vectors "
             "used in content-based similarity carry real discriminative signal. "
             "Each dimension genuinely differs between attack types.")
        ]:
            st.markdown(f"""<div class="finding-box">
                <p><strong>{title}</strong><br>{desc}</p>
            </div>""", unsafe_allow_html=True)

# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================
elif page == "Recommendation System":

    st.markdown("## DDoS Attack Type Recommendation System")
    st.markdown(
        "Given an observed DDoS attack on a network service, which other attack "
        "types should the SOC analyst prepare defences for?"
    )
    st.markdown("---")

    als_results  = als_res
    eval_results = eval_res

    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.markdown("#### Analyst Query")

        attack_options  = sorted(cf_sim.index.tolist())
        service_options = sorted(als_results.keys()) if als_results \
            else ["Other_Service"]

        observed = st.selectbox("Observed attack type", attack_options)
        service  = st.selectbox("Targeted service", service_options)
        top_k    = st.slider("Top-K recommendations", 1, 10, 5)
        alpha    = st.slider(
            "CF weight (alpha)", 0.0, 1.0, 0.6, 0.1,
            help="Higher = more weight on co-occurrence. "
                 "Lower = more weight on feature similarity."
        )
        use_llm = st.checkbox("Include LLM reflection", value=True)
        api_key = ""
        if use_llm:
            api_key = st.text_input(
                "OpenAI API key",
                type="password",
                help="Used only for this request. Not stored."
            )

        run = st.button("Get Recommendations", type="primary")

        st.markdown("---")
        st.markdown("#### Observed Attack Profile")
        content_cols = [c for c in centroids.columns if c.startswith("avg_")]
        if observed in centroids.index:
            prow = centroids.loc[observed, content_cols]
            st.dataframe(pd.DataFrame({
                "Feature"   : [c.replace("avg_", "") for c in content_cols],
                "Mean Value": [round(float(v), 4) for v in prow.values]
            }), use_container_width=True, hide_index=True)

    with col_output:
        if run:
            st.markdown("#### Recommendations")

            cf_scores = cf_sim[observed].drop(observed, errors='ignore')
            cb_scores = cb_sim[observed].drop(observed, errors='ignore')
            cf_top    = cf_scores.nlargest(top_k)
            cb_top    = cb_scores.nlargest(top_k)
            hybrid    = get_hybrid(observed, K=top_k, alpha=alpha)
            als_recs  = als_results.get(service, [])

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Hybrid — Final",
                "LLM Reflection",
                "Collaborative Filtering",
                "Content-Based",
                "ALS",
                "Evaluation"
            ])

            with tab1:
                st.markdown(
                    f"Combined recommendation for **{observed}** on **{service}**  \n"
                    f"CF weight: {alpha} | Content weight: {round(1-alpha,1)}"
                )
                if hybrid:
                    st.dataframe(pd.DataFrame({
                        "Rank"         : range(1, len(hybrid)+1),
                        "Attack Type"  : [r["attack_type"] for r in hybrid],
                        "Hybrid Score" : [r["hybrid_score"] for r in hybrid],
                        "CF Score"     : [r["cf_score"] for r in hybrid],
                        "Content Score": [r["content_score"] for r in hybrid]
                    }), use_container_width=True, hide_index=True)
                    st.markdown("**Prepare defences for:**")
                    for i, r in enumerate(hybrid, 1):
                        st.markdown(f"{i}. {r['attack_type']}")

            with tab2:
                if use_llm and api_key:
                    with st.spinner("GPT-4o-mini is reasoning about the traffic profile..."):
                        try:
                            candidates = [r["attack_type"] for r in hybrid]
                            llm_ranked = call_llm(observed, candidates, centroids, api_key)
                            st.markdown(
                                "GPT-4o-mini re-ranked hybrid candidates based on the "
                                "observed traffic profile — adding domain reasoning that "
                                "co-occurrence statistics alone cannot provide."
                            )
                            st.markdown("")
                            for item in llm_ranked:
                                st.markdown(
                                    f"""<div class="llm-card">
                                    <div class="llm-rank">{item['rank']}. {item['attack_type']}</div>
                                    <div class="llm-just">{item['justification']}</div>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                        except Exception as e:
                            st.error(f"LLM call failed: {e}")
                elif use_llm and not api_key:
                    st.warning("Enter your OpenAI API key to enable LLM reflection.")
                else:
                    st.info("LLM reflection is disabled.")

            with tab3:
                st.markdown(
                    "Attack types that co-occur with the observed attack "
                    "across similar service profiles."
                )
                st.dataframe(pd.DataFrame({
                    "Rank"         : range(1, top_k+1),
                    "Attack Type"  : cf_top.index.tolist(),
                    "CF Similarity": cf_top.values.round(4)
                }), use_container_width=True, hide_index=True)

            with tab4:
                st.markdown(
                    "Attack types with similar traffic feature profiles. "
                    "Uses centroid vectors from Part A Q2."
                )
                st.dataframe(pd.DataFrame({
                    "Rank"              : range(1, top_k+1),
                    "Attack Type"       : cb_top.index.tolist(),
                    "Feature Similarity": cb_top.values.round(4)
                }), use_container_width=True, hide_index=True)
                st.caption(
                    "Negative similarity = opposite traffic signatures — "
                    "attacks that co-occur on the same services but work "
                    "through completely different mechanisms."
                )

            with tab5:
                st.markdown(
                    f"ALS recommendations for **{service}** — latent factor "
                    f"learning from implicit flow count data."
                )
                if als_recs:
                    st.dataframe(pd.DataFrame({
                        "Rank"       : range(1, len(als_recs[:top_k])+1),
                        "Attack Type": [r[0] for r in als_recs[:top_k]],
                        "ALS Score"  : [r[1] for r in als_recs[:top_k]]
                    }), use_container_width=True, hide_index=True)
                    st.caption(
                        "ALS score is a confidence-weighted preference score. "
                        "RMSE = 859 is expected for implicit ALS on "
                        "high-variance count data (range 5 to 133,012 flows)."
                    )
                else:
                    st.info("No ALS results for this service.")

            with tab6:
                if eval_results:
                    e1, e2, e3, e4 = st.columns(4)
                    with e1:
                        st.metric("ALS RMSE", f"{eval_results.get('rmse', 0):.2f}")
                    with e2:
                        st.metric("Precision@5",
                                  f"{eval_results.get('precision_at_5', 0):.4f}")
                    with e3:
                        st.metric("Recall@5",
                                  f"{eval_results.get('recall_at_5', 0):.4f}")
                    with e4:
                        st.metric("Best Alpha",
                                  f"{eval_results.get('best_alpha', 0.6)}")

                    st.markdown("---")
                    st.markdown(
                        "**Precision@5 = 1.0** — all top-5 recommendations are "
                        "genuinely observed attack types for that service.  \n"
                        "**Recall@5 = 0.294** — top-5 covers 5 of 17 possible "
                        "attack types (5/17 = 0.294).  \n"
                        "**Alpha tuning** — all alpha values (0.3 to 0.9) gave "
                        "identical Precision@5 because the controlled dataset has "
                        "0% sparsity. In a real deployment, the optimal alpha "
                        "would be clearly identifiable from the tuning chart."
                    )
                    chart = os.path.join(BASE, 'evaluation.png')
                    if os.path.exists(chart):
                        st.image(chart, use_container_width=True)

        else:
            st.info(
                "Select an attack type and service on the left, "
                "then click Get Recommendations."
            )

        st.markdown("---")
        st.markdown("#### How the System Works")
        st.markdown("""
| Layer | Method | What it does |
|---|---|---|
| 1 | Collaborative Filtering (Item-KNN) | Finds attack types that historically co-occur across similar network services |
| 2 | ALS Matrix Factorization | Learns hidden patterns between services and attack types from flow count data |
| 3 | Content-Based Similarity | Finds attack types with similar traffic signatures using Part A feature centroids |
| 4 | Hybrid Combination | Combines co-occurrence and feature similarity into one ranked list |
| 5 | LLM Reflection | Re-ranks candidates using traffic profile reasoning — which attacks are operationally plausible |
""")