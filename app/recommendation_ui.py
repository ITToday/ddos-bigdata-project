# ============================================================
# DDoS Big Data Analytics & Recommendation System
# Streamlit application — reads precomputed Spark outputs
# University of Moratuwa — MSc AI — Big Data Analytics
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

# --- Clean minimal styling ---
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; }
    h1 { font-size: 22px; font-weight: 600; color: #1a1a2e; }
    h2 { font-size: 16px; font-weight: 600; color: #1a1a2e; }
    h3 { font-size: 14px; font-weight: 500; color: #374151; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 2px;
    }
    .finding-box {
        border-left: 3px solid #1a1a2e;
        padding: 8px 14px;
        margin-bottom: 10px;
        background: #f8f9fa;
    }
    .finding-box p {
        margin: 0;
        font-size: 13px;
        color: #374151;
    }
    .section-label {
        font-size: 11px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Paths ---
BASE = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DISTRIBUTION_PATH   = os.path.join(BASE, 'distribution.csv')
CENTROIDS_PATH      = os.path.join(BASE, 'attack_type_centroids.parquet')
MATRIX_PATH         = os.path.join(BASE, 'interaction_matrix.csv')
DISCRIMINATION_PATH = os.path.join(BASE, 'discrimination.json')

# --- Data loading ---
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
        data = json.load(f)
    return pd.DataFrame(data)

@st.cache_data
def compute_cf_similarity(matrix_pd):
    vectors = matrix_pd.values.astype(float)
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalised = vectors / norms
    sim = np.dot(normalised, normalised.T)
    return pd.DataFrame(sim, index=matrix_pd.index, columns=matrix_pd.index)

@st.cache_data
def compute_content_similarity(centroids_pd):
    content_cols = [c for c in centroids_pd.columns if c.startswith("avg_")]
    content_mat  = centroids_pd[content_cols].fillna(0)
    scaler       = StandardScaler()
    scaled       = scaler.fit_transform(content_mat.values)
    norms        = np.linalg.norm(scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalised   = scaled / norms
    sim          = np.dot(normalised, normalised.T)
    return pd.DataFrame(sim, index=content_mat.index, columns=content_mat.index)

try:
    dist_df   = load_distribution()
    centroids = load_centroids().set_index("activity")
    matrix_pd = load_interaction_matrix()
    disc_df   = load_discrimination()
    cf_sim    = compute_cf_similarity(matrix_pd)
    cb_sim    = compute_content_similarity(centroids)
    data_ok   = True
except Exception as e:
    data_ok  = False
    data_err = str(e)

# --- Sidebar ---
st.sidebar.markdown("## DDoS Analytics Platform")
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
    atk_types = len(dist_df[dist_df['label'] == 'Attack'])
    ben_cats  = len(dist_df[dist_df['label'] == 'Benign'])
    st.sidebar.markdown("**Data status**")
    st.sidebar.markdown(f"- {atk_types} attack types loaded")
    st.sidebar.markdown(f"- {ben_cats} benign categories")
    st.sidebar.markdown(f"- {len(centroids)} centroid vectors")
else:
    st.sidebar.error("Output files not found. Run Part A notebook first.")
    st.stop()

# ============================================================
# OVERVIEW PAGE
# ============================================================
if page == "Overview":

    st.markdown("## DDoS Network Traffic — Big Data Analytics")
    st.markdown(
        "Analysis of 540,494 DDoS network flow records across 319 features "
        "using Apache Spark. Five analytical questions answered."
    )
    st.markdown("---")

    total  = int(dist_df['flow_count'].sum())
    atk    = int(dist_df[dist_df['label'] == 'Attack']['flow_count'].sum())
    ben    = int(dist_df[dist_df['label'] == 'Benign']['flow_count'].sum())
    sus    = int(dist_df[dist_df['label'] == 'Suspicious']['flow_count'].sum())

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
            <div class="metric-label">Attack flows</div>
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

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Key Findings")
        findings = [
            ("Two-level classification",
             "The dataset uses label (3 categories) and activity "
             "(26 specific scenarios). All analysis groups by activity."),
            ("Severe class imbalance",
             "Attack-TCP-BYPass-V1 accounts for 78.6% of attack flows. "
             "The remaining 16 types have 625–3,147 flows each."),
            ("Distinct attack signatures",
             "Each attack type has measurable differences in bytes rate, "
             "packet rate, duration, and TCP flag combinations."),
            ("BYPass mechanism",
             "Lowest packet rate (61.55/sec) but highest flow count. "
             "Operates through connection state exhaustion."),
            ("Top discriminating features",
             "fwd_init_win_bytes (ratio 4.12) and packets_IAT_mean "
             "(ratio 1.75) most clearly separate attack from benign.")
        ]
        for title, desc in findings:
            st.markdown(f"""<div class="finding-box">
                <p><strong>{title}</strong><br>{desc}</p>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("#### Analytical Pipeline")
        pipeline = pd.DataFrame([
            ["Q1", "Attack type distribution",
             "GROUP BY + SUM() OVER() window"],
            ["Q2", "Feature signatures per attack",
             "groupBy().agg() programmatic"],
            ["Q3", "Timing patterns + intensity ranking",
             "RANK() OVER() window function"],
            ["Q4", "TCP flag combinations",
             "AVG() grouped by activity"],
            ["Q5", "Feature discrimination",
             "Mean ratio Attack vs Benign"]
        ], columns=["Section", "Question", "Spark Technique"])
        st.dataframe(pipeline, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Dataset")
        info = pd.DataFrame([
            ["Source", "York University / cPacket Networks"],
            ["Published", "March 2024"],
            ["License", "CC-BY-SA-4.0"],
            ["Rows", f"{total:,}"],
            ["Features", "319"],
            ["Format", "Parquet"],
            ["Engine", "Apache Spark 4.0 / PySpark"]
        ], columns=["", ""])
        st.dataframe(info, use_container_width=True, hide_index=True)

# ============================================================
# Q1 PAGE
# ============================================================
elif page == "Q1 — Attack Distribution":

    st.markdown("## Q1 — Attack Type Distribution")
    st.markdown(
        "What is the distribution of specific DDoS attack types? "
        "Which are dominant and which are rare?"
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
    table['pct_of_total'] = (
        table['flow_count'] / table['flow_count'].sum() * 100
    ).round(2)

    st.dataframe(
        table[['activity','label','flow_count','pct_of_total']],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("#### Finding")
    st.markdown("""<div class="finding-box"><p>
        Attack-TCP-BYPass-V1 dominates at 78.6% of all attack flows.
        The remaining 16 attack types have 625–3,147 flows each.
        In a SOC, these minority classes are most dangerous — their rarity
        means fewer alerts and higher chance of evading threshold-based
        detection rules. This motivates the hybrid recommendation approach
        in Part B where content-based similarity complements CF to surface
        rare attack types.
    </p></div>""", unsafe_allow_html=True)

# ============================================================
# Q2 PAGE
# ============================================================
elif page == "Q2 — Feature Signatures":

    st.markdown("## Q2 — Traffic Feature Signatures")
    st.markdown(
        "What are the measurable traffic characteristics that define "
        "each specific attack type?"
    )
    st.markdown("---")

    attack_types = sorted(centroids.index.tolist())
    selected = st.selectbox("Select attack type to inspect:", attack_types)

    if selected:
        content_cols = [c for c in centroids.columns if c.startswith("avg_")]
        row = centroids.loc[selected, content_cols]

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Feature Profile")
            profile = pd.DataFrame({
                "Feature": [c.replace("avg_","") for c in content_cols],
                "Value":   [round(float(v), 4) for v in row.values]
            })
            st.dataframe(profile, use_container_width=True, hide_index=True)

        with col_r:
            st.markdown("#### Comparison — All Attack Types")
            all_profiles = centroids[content_cols].copy()
            all_profiles.columns = [
                c.replace("avg_","") for c in content_cols
            ]
            st.dataframe(
                all_profiles.round(4),
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("#### Finding")
    st.markdown("""<div class="finding-box"><p>
        Nine features were selected to cover each meaningful traffic dimension
        without redundancy — volume (bytes_rate), intensity (packets_rate),
        timing (duration), payload (payload_bytes_mean), TCP flags (syn, ack),
        asymmetry (down_up_rate), and directional rates (fwd, bwd).
        These centroids become the content vectors for Part B
        content-based similarity.
    </p></div>""", unsafe_allow_html=True)

# ============================================================
# Q3 PAGE
# ============================================================
elif page == "Q3 — Timing Patterns":

    st.markdown("## Q3 — Flow Timing Pattern Analysis")
    st.markdown(
        "How do packet inter-arrival times and flow durations differ "
        "across specific attack types?"
    )
    st.markdown("---")

    chart = os.path.join(BASE, 'q3_timing_patterns.png')
    if os.path.exists(chart):
        st.image(chart, use_container_width=True)
    else:
        st.warning("Chart not found. Run Part A notebook first.")

    st.markdown("---")
    st.markdown("#### Finding")
    st.markdown("""<div class="finding-box"><p>
        Attack-TCP-BYPass-V1 ranks last in packet rate (61.55/sec) but has
        the highest flow count (134,110). Each flow lasts 0.0025 seconds with
        1.02 packets on average — connection state exhaustion rather than
        bandwidth flooding. High-intensity volumetric attacks (Flag-OSYNP at
        12,149 pkt/sec) show the opposite profile — high rate, short duration.
        The RANK() OVER window function computes this ranking across all
        activity types in a single query without collapsing rows.
    </p></div>""", unsafe_allow_html=True)

# ============================================================
# Q4 PAGE
# ============================================================
elif page == "Q4 — Flag Signatures":

    st.markdown("## Q4 — TCP Flag Signature Analysis")
    st.markdown(
        "Which TCP flag combinations are characteristic of each attack type?"
    )
    st.markdown("---")

    chart = os.path.join(BASE, 'q4_flag_heatmap.png')
    if os.path.exists(chart):
        st.image(chart, use_container_width=True)
    else:
        st.warning("Chart not found. Run Part A notebook first.")

    st.markdown("---")
    st.markdown("#### Flag Reference")
    flags = pd.DataFrame([
        ["SYN", "Synchronise",
         "SYN floods — initiate half-open connections to exhaust server state"],
        ["ACK", "Acknowledge",
         "ACK floods — send fake established-session traffic"],
        ["PSH", "Push",
         "Combined with ACK in application-layer floods"],
        ["FIN", "Finish",
         "FIN floods — abuse connection teardown"],
        ["RST", "Reset",
         "RST attacks — force connection termination"],
        ["URG", "Urgent",
         "Rarely used — presence in attack traffic is itself anomalous"]
    ], columns=["Flag", "Full Name", "Attack Relevance"])
    st.dataframe(flags, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Finding")
    st.markdown("""<div class="finding-box"><p>
        The clearest separator between attack and benign traffic is the
        combination of high SYN with suppressed ACK and suppressed FIN.
        Legitimate TCP always produces ACK and FIN as sessions are established
        and closed cleanly. Attack flows abort without clean teardown.
        Mixed-flag attacks (TCP-Control, TCP-Flag-MIX) deliberately vary
        flag patterns to evade single-flag threshold rules in a SIEM.
    </p></div>""", unsafe_allow_html=True)

# ============================================================
# Q5 PAGE
# ============================================================
elif page == "Q5 — Discrimination":

    st.markdown("## Q5 — Feature Discrimination Analysis")
    st.markdown(
        "Which features most clearly separate attack from benign traffic?"
    )
    st.markdown("---")

    chart = os.path.join(BASE, 'q5_discrimination.png')
    if os.path.exists(chart):
        st.image(chart, use_container_width=True)
    else:
        st.warning("Chart not found. Run Part A notebook first.")

    st.markdown("---")
    st.markdown("#### Discrimination Ratios")

    disc_table = disc_df.copy()
    disc_table.columns = ["Feature", "Ratio", "Direction"]
    disc_table["Direction"] = disc_table["Direction"].map({
        "attack": "Attack higher",
        "benign": "Benign higher"
    })
    st.dataframe(
        disc_table.sort_values("Ratio", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("#### Finding")
    st.markdown("""<div class="finding-box"><p>
        fwd_init_win_bytes (ratio 4.12) and packets_IAT_mean (ratio 1.75)
        are the only features where attack traffic is higher than benign.
        All remaining features show ratios below 1.0 — benign traffic is
        higher due to high-bandwidth protocols like FTP and web browsing.
        These 14 features validate the content vectors used in Part B —
        each dimension carries real discriminative signal.
    </p></div>""", unsafe_allow_html=True)

# ============================================================
# RECOMMENDATION SYSTEM PAGE
# ============================================================
elif page == "Recommendation System":

    st.markdown("## DDoS Attack Type Recommendation System")
    st.markdown(
        "Given an observed DDoS attack on a network service, the system "
        "recommends which other attack types to prepare defences for. "
        "Four layers: collaborative filtering, ALS, content-based "
        "similarity, and LLM reflection (CRAG-motivated)."
    )
    st.markdown("---")

    # --- Load Part B outputs ---
    @st.cache_data
    def load_als_results():
        path = os.path.join(BASE, 'als_results.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    @st.cache_data
    def load_eval_results():
        path = os.path.join(BASE, 'evaluation_results.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    als_results  = load_als_results()
    eval_results = load_eval_results()

    # --- LLM function ---
    def build_attack_profile(query_attack, centroids_pd):
        if query_attack not in centroids_pd.index:
            return f"Attack type: {query_attack} (profile unavailable)"
        row = centroids_pd.loc[query_attack]
        feature_labels = {
            "avg_bytes_rate"        : "Avg bytes rate",
            "avg_packets_rate"      : "Avg packets/sec",
            "avg_duration"          : "Avg flow duration (s)",
            "avg_payload_bytes_mean": "Avg payload size (bytes)",
            "avg_syn_flag_counts"   : "Avg SYN flag count",
            "avg_ack_flag_counts"   : "Avg ACK flag count",
            "avg_down_up_rate"      : "Down/up traffic ratio",
            "avg_fwd_packets_rate"  : "Forward packets/sec",
            "avg_bwd_packets_rate"  : "Backward packets/sec"
        }
        lines = [f"Observed attack type: {query_attack}"]
        for col, label in feature_labels.items():
            if col in row.index:
                lines.append(f"  {label}: {row[col]:.4f}")
        return "\n".join(lines)

    def llm_rerank(query_attack, candidates, centroids_pd, api_key):
        from openai import OpenAI
        client  = OpenAI(api_key=api_key)
        profile = build_attack_profile(query_attack, centroids_pd)
        cand_str = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(candidates)
        )
        prompt = f"""You are a cybersecurity expert specialising in DDoS attack analysis.

OBSERVED ATTACK PROFILE:
{profile}

HYBRID RECOMMENDATION CANDIDATES:
{cand_str}

TASK:
Re-rank these candidates from most to least relevant for a SOC analyst
to prepare defences against, based on the observed traffic features.
Consider packet rate, flow duration, flag patterns, and bytes rate.

Return ONLY a JSON object, no other text:
{{
  "ranked_recommendations": [
    {{"rank": 1, "attack_type": "...", "justification": "one sentence"}},
    {{"rank": 2, "attack_type": "...", "justification": "one sentence"}},
    {{"rank": 3, "attack_type": "...", "justification": "one sentence"}},
    {{"rank": 4, "attack_type": "...", "justification": "one sentence"}},
    {{"rank": 5, "attack_type": "...", "justification": "one sentence"}}
  ]
}}"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600
        )
        raw = response.choices[0].message.content.strip()
        try:
            return json.loads(raw)["ranked_recommendations"]
        except:
            clean = raw.replace("```json","").replace("```","").strip()
            return json.loads(clean)["ranked_recommendations"]

    # --- Layout ---
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### Analyst Query")

        attack_options  = sorted(cf_sim.index.tolist())
        service_options = sorted(als_results.keys()) if als_results \
            else ["Other_Service"]

        observed = st.selectbox("Observed attack type", attack_options)
        service  = st.selectbox("Targeted service", service_options)
        top_k    = st.slider("Top-K", 1, 10, 5)
        alpha    = st.slider(
            "CF weight (alpha)", 0.0, 1.0, 0.6, 0.1,
            help="Weight for CF score. 1-alpha = content weight."
        )

        use_llm = st.checkbox("Include LLM reflection (GPT-4o-mini)", value=True)

        api_key = ""
        if use_llm:
            api_key = st.text_input(
                "OpenAI API key",
                type="password",
                help="Key is used only for this request. Not stored."
            )

        run = st.button("Get Recommendations", type="primary")

        st.markdown("---")
        st.markdown("#### Traffic Profile")
        content_cols = [c for c in centroids.columns if c.startswith("avg_")]
        if observed in centroids.index:
            profile = centroids.loc[observed, content_cols]
            profile_df = pd.DataFrame({
                "Feature": [c.replace("avg_","") for c in content_cols],
                "Value"  : [round(float(v), 4) for v in profile.values]
            })
            st.dataframe(profile_df, use_container_width=True, hide_index=True)

    with col_right:
        if run:
            st.markdown("#### Recommendations")

            # --- CF ---
            cf_scores = cf_sim[observed].drop(observed, errors='ignore')
            cf_top    = cf_scores.nlargest(top_k)

            # --- Content ---
            cb_scores = cb_sim[observed].drop(observed, errors='ignore')
            cb_top    = cb_scores.nlargest(top_k)

            # --- Hybrid ---
            common = cf_scores.index.intersection(cb_scores.index)
            cf_n   = (cf_scores[common] - cf_scores[common].min()) / \
                     (cf_scores[common].max() - cf_scores[common].min() + 1e-9)
            cb_n   = (cb_scores[common] - cb_scores[common].min()) / \
                     (cb_scores[common].max() - cb_scores[common].min() + 1e-9)
            hybrid = (alpha * cf_n + (1-alpha) * cb_n).sort_values(
                ascending=False
            )

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Final (Hybrid)",
                "LLM Reflection",
                "Collaborative Filtering",
                "Content-Based",
                "Evaluation"
            ])

            with tab1:
                st.markdown(
                    f"Hybrid recommendations for **{observed}** "
                    f"(CF={alpha}, Content={round(1-alpha,1)})"
                )
                hybrid_df = pd.DataFrame({
                    "Rank"         : range(1, top_k+1),
                    "Attack Type"  : hybrid.head(top_k).index.tolist(),
                    "Hybrid Score" : hybrid.head(top_k).values.round(4),
                    "CF Score"     : cf_n[hybrid.head(top_k).index].round(4),
                    "Content Score": cb_n[hybrid.head(top_k).index].round(4)
                })
                st.dataframe(hybrid_df, use_container_width=True,
                             hide_index=True)

            with tab2:
                if use_llm and api_key:
                    with st.spinner("Calling GPT-4o-mini..."):
                        try:
                            candidates = hybrid.head(top_k).index.tolist()
                            llm_ranked = llm_rerank(
                                observed, candidates, centroids, api_key
                            )
                            st.markdown(
                                "GPT-4o-mini re-ranked the hybrid candidates "
                                "using observed traffic profile reasoning "
                                "(CRAG-motivated)."
                            )
                            for item in llm_ranked:
                                st.markdown(
                                    f"**{item['rank']}. {item['attack_type']}**"
                                )
                                st.markdown(
                                    f"*{item['justification']}*"
                                )
                                st.markdown("---")
                        except Exception as e:
                            st.error(f"LLM call failed: {e}")
                elif use_llm and not api_key:
                    st.warning(
                        "Enter your OpenAI API key in the left panel "
                        "to enable LLM reflection."
                    )
                else:
                    st.info("LLM reflection is disabled.")

            with tab3:
                st.markdown(
                    "Attack types that co-occur with the observed attack "
                    "across similar service profiles."
                )
                cf_df = pd.DataFrame({
                    "Rank"         : range(1, top_k+1),
                    "Attack Type"  : cf_top.index.tolist(),
                    "CF Similarity": cf_top.values.round(4)
                })
                st.dataframe(cf_df, use_container_width=True, hide_index=True)

            with tab4:
                st.markdown(
                    "Attack types with similar traffic feature profiles "
                    "to the observed attack (centroids from Part A)."
                )
                cb_df = pd.DataFrame({
                    "Rank"              : range(1, top_k+1),
                    "Attack Type"       : cb_top.index.tolist(),
                    "Feature Similarity": cb_top.values.round(4)
                })
                st.dataframe(cb_df, use_container_width=True, hide_index=True)

            with tab5:
                if eval_results:
                    e1, e2, e3 = st.columns(3)
                    with e1:
                        st.metric("ALS RMSE", f"{eval_results.get('rmse', 0):.2f}")
                    with e2:
                        st.metric("Precision@5",
                                  f"{eval_results.get('precision_at_5', 0):.4f}")
                    with e3:
                        st.metric("Recall@5",
                                  f"{eval_results.get('recall_at_5', 0):.4f}")

                    st.markdown("---")
                    st.markdown("#### Alpha Tuning Results")
                    alpha_data = pd.DataFrame([
                        {"Alpha": k, "Precision@5": v}
                        for k, v in eval_results.get(
                            "alpha_results", {}
                        ).items()
                    ])
                    if not alpha_data.empty:
                        st.dataframe(
                            alpha_data, use_container_width=True,
                            hide_index=True
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
        st.markdown("#### System Architecture")
        st.markdown("""
        | Layer | Method | Course Reference |
        |---|---|---|
        | Layer 1 | Collaborative Filtering (Item-KNN) | CF lecture, movie recommendation practical |
        | Layer 2 | ALS Matrix Factorization | CF lecture, pyspark.ml.recommendation |
        | Layer 3 | Content-Based Similarity | Word similarity practical |
        | Layer 4 | Hybrid Combination | T6 handout — alpha weighted |
        | Layer 5 | LLM Reflection | CRAG (Zhu et al., WWW 2025) |
        """)