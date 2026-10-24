import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SUMO RL Traffic Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

SUMO_DIR = os.path.join(
    PROJECT_DIR,
    "sumo"
)

SUMO_CONFIG = os.path.join(
    SUMO_DIR,
    "RL.sumocfg"
)

RL_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "rl_results.csv"
)

RL_EVALUATION_FILE = os.path.join(
    RESULTS_DIR,
    "rl_evaluation.csv"
)

BASELINE_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "baseline_results.csv"
)

COMPARISON_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "comparison_results.csv"
)

EVOLUTION_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "evolution_results.csv"
)

TRAFFIC_DATA_FILE = os.path.join(
    RESULTS_DIR,
    "traffic_data.csv"
)

Q_TABLE_FILE = os.path.join(
    RESULTS_DIR,
    "q_table.pkl"
)

EVOLUTION_POLICY_FILE = os.path.join(
    RESULTS_DIR,
    "best_evolution_policy.csv"
)


# ============================================================
# LOGIN
# ============================================================

VALID_USERNAME = "adminme_26"
VALID_PASSWORD = "802632"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #888888;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .login-box {
        max-width: 450px;
        margin: 100px auto;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        '<div class="login-box">',
        unsafe_allow_html=True
    )

    st.title("🚦 SUMO Traffic Optimization")

    st.markdown("### Login")

    st.write(
        "Enter the authorized credentials to access the dashboard."
    )

    username = st.text_input(
        "Username",
        placeholder="Enter username"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password"
    )

    if st.button(
        "Login",
        use_container_width=True
    ):

        if (
            username == VALID_USERNAME
            and password == VALID_PASSWORD
        ):

            st.session_state.logged_in = True
            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=5)
def load_rl_results():

    if not os.path.exists(RL_RESULTS_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(RL_RESULTS_FILE)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_rl_evaluation():

    if not os.path.exists(RL_EVALUATION_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(RL_EVALUATION_FILE)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_baseline_results():

    if not os.path.exists(BASELINE_RESULTS_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(BASELINE_RESULTS_FILE)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_comparison_results():

    if not os.path.exists(COMPARISON_RESULTS_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(COMPARISON_RESULTS_FILE)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_evolution_results():

    if not os.path.exists(EVOLUTION_RESULTS_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(EVOLUTION_RESULTS_FILE)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_traffic_data():

    if not os.path.exists(TRAFFIC_DATA_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(TRAFFIC_DATA_FILE)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_evolution_policy():

    if not os.path.exists(EVOLUTION_POLICY_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(EVOLUTION_POLICY_FILE)
    except Exception:
        return pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():

    st.sidebar.title("🚦 SUMO Control")

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "RL Training",
            "Evolutionary Optimization",
            "Comparison",
            "Traffic Analysis",
            "About"
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("Project")

    st.sidebar.write(
        "SUMO-RL Traffic Signal Optimization"
    )

    st.sidebar.caption(
        f"Config: {os.path.basename(SUMO_CONFIG)}"
    )

    st.sidebar.markdown("---")

    if st.sidebar.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.rerun()

    return page


# ============================================================
# OVERVIEW
# ============================================================

def overview():

    st.markdown(
        '<div class="main-title">🚦 Adaptive Traffic Signal Control</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Traffic signal optimization using reinforcement learning and evolutionary optimization</div>',
        unsafe_allow_html=True
    )

    rl_data = load_rl_results()
    evolution_data = load_evolution_results()
    comparison_data = load_comparison_results()
    traffic_data = load_traffic_data()

    # --------------------------------------------------------
    # PROJECT STATUS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "RL Training",
            "Available"
            if not rl_data.empty
            else "Not Available"
        )

    with col2:

        st.metric(
            "Evolutionary",
            "Available"
            if not evolution_data.empty
            else "Not Available"
        )

    with col3:

        st.metric(
            "Traffic Records",
            f"{len(traffic_data):,}"
        )

    with col4:

        if not evolution_data.empty:

            best_fitness = evolution_data[
                "Best_Fitness"
            ].min()

            st.metric(
                "Best Fitness",
                f"{best_fitness:.3f}"
            )

        else:

            st.metric(
                "Best Fitness",
                "N/A"
            )

    st.markdown("---")

    # --------------------------------------------------------
    # COMPARISON SUMMARY
    # --------------------------------------------------------

    if not comparison_data.empty:

        st.subheader(
            "RL vs Fixed-Time Baseline"
        )

        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

    # --------------------------------------------------------
    # EVOLUTION SUMMARY
    # --------------------------------------------------------

    if not evolution_data.empty:

        st.subheader(
            "Evolutionary Optimization Results"
        )

        best_row = evolution_data.loc[
            evolution_data["Best_Fitness"].idxmin()
        ]

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Best Fitness",
                f"{best_row['Best_Fitness']:.3f}"
            )

        with c2:

            st.metric(
                "Best Waiting Time",
                f"{best_row['Best_Waiting_Time']:.2f}"
            )

        with c3:

            st.metric(
                "Best Queue",
                f"{best_row['Best_Queue']:.2f}"
            )

        with c4:

            st.metric(
                "Best Throughput",
                f"{best_row['Best_Throughput']:.0f}"
            )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            if "Best_Fitness" in evolution_data.columns:

                fig = px.line(
                    evolution_data,
                    x="Generation",
                    y="Best_Fitness",
                    markers=True,
                    title="Best Fitness by Generation"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with col2:

            if "Best_Waiting_Time" in evolution_data.columns:

                fig = px.line(
                    evolution_data,
                    x="Generation",
                    y="Best_Waiting_Time",
                    markers=True,
                    title="Best Waiting Time by Generation"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    else:

        st.info(
            "No evolutionary optimization results found."
        )


# ============================================================
# RL RESULTS PAGE
# ============================================================

def rl_training_page():

    st.title(
        "🤖 Q-Learning Results"
    )

    st.write(
        "Results from the Q-learning traffic signal controller."
    )

    st.markdown("---")

    rl_data = load_rl_results()
    evaluation_data = load_rl_evaluation()

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Episodes",
            "100"
        )

    with col2:

        st.metric(
            "Steps / Episode",
            "500"
        )

    with col3:

        st.metric(
            "Total Steps",
            "50,000"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if rl_data.empty:

        st.warning(
            "No Q-learning results available."
        )

        return

    # --------------------------------------------------------
    # LATEST TRAINING RESULTS
    # --------------------------------------------------------

    st.subheader(
        "Training Results"
    )

    latest = rl_data.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Latest Reward",
            f"{latest['Total_Reward']:.2f}"
        )

    with c2:

        st.metric(
            "Steps",
            int(latest["Steps"])
        )

    with c3:

        st.metric(
            "Q-table States",
            int(latest["Q_Table_States"])
        )

    with c4:

        if "Signal_Changes" in rl_data.columns:

            st.metric(
                "Signal Changes",
                int(latest["Signal_Changes"])
            )

        else:

            st.metric(
                "Signal Changes",
                "N/A"
            )

    st.markdown("---")

    # --------------------------------------------------------
    # REWARD
    # --------------------------------------------------------

    if "Episode" in rl_data.columns:

        fig = px.line(
            rl_data,
            x="Episode",
            y="Total_Reward",
            markers=True,
            title="Q-Learning Reward per Episode"
        )

        fig.update_layout(
            xaxis_title="Episode",
            yaxis_title="Total Reward"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # EPSILON
    # --------------------------------------------------------

    if "Epsilon" in rl_data.columns:

        fig = px.line(
            rl_data,
            x="Episode",
            y="Epsilon",
            markers=True,
            title="Exploration Rate (Epsilon)"
        )

        fig.update_layout(
            xaxis_title="Episode",
            yaxis_title="Epsilon"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Q-TABLE GROWTH
    # --------------------------------------------------------

    if "Q_Table_States" in rl_data.columns:

        fig = px.line(
            rl_data,
            x="Episode",
            y="Q_Table_States",
            markers=True,
            title="Q-Table State Growth"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SIGNAL CHANGES
    # --------------------------------------------------------

    if "Signal_Changes" in rl_data.columns:

        fig = px.line(
            rl_data,
            x="Episode",
            y="Signal_Changes",
            markers=True,
            title="Traffic Signal Changes"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # EVALUATION RESULTS
    # --------------------------------------------------------

    if not evaluation_data.empty:

        st.markdown("---")

        st.subheader(
            "RL Evaluation"
        )

        eval_latest = evaluation_data.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Average Waiting Time",
                f"{eval_latest['Average_Waiting_Time']:.3f}"
            )

        with c2:

            st.metric(
                "Average Queue",
                f"{eval_latest['Average_Queue']:.3f}"
            )

        with c3:

            st.metric(
                "Average Vehicles",
                f"{eval_latest['Average_Vehicles']:.3f}"
            )

        with c4:

            st.metric(
                "Throughput",
                f"{eval_latest['Throughput']:.3f}"
            )

        st.dataframe(
            evaluation_data,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # FULL TRAINING DATA
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Episode Results"
    )

    st.dataframe(
        rl_data,
        use_container_width=True,
        hide_index=True
    )

    csv_data = rl_data.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download RL Results",
        csv_data,
        "rl_results.csv",
        "text/csv"
    )


# ============================================================
# EVOLUTIONARY OPTIMIZATION
# ============================================================

def evolutionary_page():

    st.title(
        "🧬 Evolutionary Optimization"
    )

    st.write(
        "Evolutionary optimization results for traffic signal phase durations."
    )

    st.markdown("---")

    evolution_data = load_evolution_results()

    if evolution_data.empty:

        st.warning(
            "No evolutionary optimization results found."
        )

        return

    # --------------------------------------------------------
    # BEST RESULT
    # --------------------------------------------------------

    best_row = evolution_data.loc[
        evolution_data["Best_Fitness"].idxmin()
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Best Fitness",
            f"{best_row['Best_Fitness']:.3f}"
        )

    with c2:

        st.metric(
            "Waiting Time",
            f"{best_row['Best_Waiting_Time']:.2f}"
        )

    with c3:

        st.metric(
            "Queue",
            f"{best_row['Best_Queue']:.2f}"
        )

    with c4:

        st.metric(
            "Throughput",
            f"{best_row['Best_Throughput']:.0f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PHASE DURATIONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if "Phase_1_Duration" in evolution_data.columns:

            fig = px.line(
                evolution_data,
                x="Generation",
                y="Phase_1_Duration",
                markers=True,
                title="Phase 1 Duration"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        if "Phase_2_Duration" in evolution_data.columns:

            fig = px.line(
                evolution_data,
                x="Generation",
                y="Phase_2_Duration",
                markers=True,
                title="Phase 2 Duration"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # FITNESS
    # --------------------------------------------------------

    if all(
        col in evolution_data.columns
        for col in [
            "Best_Fitness",
            "Average_Fitness"
        ]
    ):

        fig = px.line(
            evolution_data,
            x="Generation",
            y=[
                "Best_Fitness",
                "Average_Fitness"
            ],
            markers=True,
            title="Evolutionary Fitness Progress"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # WAITING TIME / QUEUE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if "Best_Waiting_Time" in evolution_data.columns:

            fig = px.line(
                evolution_data,
                x="Generation",
                y="Best_Waiting_Time",
                markers=True,
                title="Best Waiting Time"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        if "Best_Queue" in evolution_data.columns:

            fig = px.line(
                evolution_data,
                x="Generation",
                y="Best_Queue",
                markers=True,
                title="Best Queue"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # GENERATION DATA
    # --------------------------------------------------------

    st.subheader(
        "Generation Results"
    )

    st.dataframe(
        evolution_data,
        use_container_width=True,
        hide_index=True
    )

    csv_data = evolution_data.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download Evolution Results",
        csv_data,
        "evolution_results.csv",
        "text/csv"
    )

    # --------------------------------------------------------
    # BEST POLICY
    # --------------------------------------------------------

    policy_data = load_evolution_policy()

    if not policy_data.empty:

        st.markdown("---")

        st.subheader(
            "Best Evolutionary Policy"
        )

        st.dataframe(
            policy_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# COMPARISON
# ============================================================

def comparison_page():

    st.title(
        "📈 RL vs Fixed-Time Comparison"
    )

    st.write(
        "Performance comparison between the fixed-time baseline and Q-learning."
    )

    st.markdown("---")

    comparison_data = load_comparison_results()

    if comparison_data.empty:

        st.warning(
            "No comparison results available."
        )

        return

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "Performance Comparison"
    )

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # IMPROVEMENT METRICS
    # --------------------------------------------------------

    if "Improvement_Percent" in comparison_data.columns:

        fig = px.bar(
            comparison_data,
            x="Metric",
            y="Improvement_Percent",
            text="Improvement_Percent",
            title="RL Improvement over Fixed-Time Baseline"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Improvement (%)",
            xaxis_title="Metric"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # DIRECT COMPARISON
    # --------------------------------------------------------

    required_columns = [
        "Metric",
        "Fixed_Time",
        "RL"
    ]

    if all(
        col in comparison_data.columns
        for col in required_columns
    ):

        st.subheader(
            "Fixed-Time vs RL"
        )

        chart_data = comparison_data.melt(
            id_vars=["Metric"],
            value_vars=[
                "Fixed_Time",
                "RL"
            ],
            var_name="Method",
            value_name="Value"
        )

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Value",
            color="Method",
            barmode="group",
            title="Performance Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = comparison_data.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download Comparison Results",
        csv_data,
        "comparison_results.csv",
        "text/csv"
    )


# ============================================================
# TRAFFIC ANALYSIS
# ============================================================

def traffic_analysis():

    st.title(
        "📊 Traffic Analysis"
    )

    traffic_data = load_traffic_data()

    if traffic_data.empty:

        st.warning(
            "No traffic dataset available."
        )

        return

    st.write(
        f"Total records: {len(traffic_data):,}"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # LATEST TRAFFIC VALUES
    # --------------------------------------------------------

    latest = traffic_data.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Vehicles",
            f"{latest['vehicles']:.0f}"
        )

    with c2:

        st.metric(
            "Queue",
            f"{latest['queue']:.2f}"
        )

    with c3:

        st.metric(
            "Waiting Time",
            f"{latest['waiting_time']:.2f}"
        )

    with c4:

        st.metric(
            "Throughput",
            f"{latest['throughput']:.0f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # WAITING TIME
    # --------------------------------------------------------

    if "step" in traffic_data.columns:

        fig = px.line(
            traffic_data,
            x="step",
            y="waiting_time",
            title="Waiting Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # QUEUE
    # --------------------------------------------------------

    if "step" in traffic_data.columns:

        fig = px.line(
            traffic_data,
            x="step",
            y="queue",
            title="Queue Length"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    if "average_speed" in traffic_data.columns:

        fig = px.line(
            traffic_data,
            x="step",
            y="average_speed",
            title="Average Speed"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # THROUGHPUT
    # --------------------------------------------------------

    if "throughput" in traffic_data.columns:

        fig = px.line(
            traffic_data,
            x="step",
            y="throughput",
            title="Traffic Throughput"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PHASE
    # --------------------------------------------------------

    if "phase" in traffic_data.columns:

        fig = px.line(
            traffic_data,
            x="step",
            y="phase",
            title="Traffic Signal Phase"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    st.subheader(
        "Traffic Dataset"
    )

    st.dataframe(
        traffic_data.tail(500),
        use_container_width=True,
        hide_index=True
    )

    csv_data = traffic_data.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download Traffic Dataset",
        csv_data,
        "traffic_data.csv",
        "text/csv"
    )


# ============================================================
# ABOUT
# ============================================================

def about():

    st.title(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ## SUMO Traffic Signal Optimization

        This project uses the **Simulation of Urban Mobility (SUMO)**
        environment to study adaptive traffic signal control.

        ### Methods

        - Q-Learning
        - Evolutionary Optimization
        - Fixed-Time Baseline
        - Traffic Simulation
        - Performance Analysis

        ### Traffic Metrics

        - Queue length
        - Waiting time
        - Average speed
        - Throughput
        - Number of vehicles
        - Traffic signal phase

        ### Project Files

        `training.py`  
        Q-Learning training.

        `baseline.py`  
        Fixed-time baseline simulation.

        `evolution.py`  
        Evolutionary optimization.

        `sumo_runner.py`  
        SUMO / TraCI simulation interface.

        `rl_agent.py`  
        Q-Learning agent.

        `results/`  
        Generated training and traffic results.
        """
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

def dashboard():

    page = sidebar()

    if page == "Overview":

        overview()

    elif page == "RL Training":

        rl_training_page()

    elif page == "Evolutionary Optimization":

        evolutionary_page()

    elif page == "Comparison":

        comparison_page()

    elif page == "Traffic Analysis":

        traffic_analysis()

    elif page == "About":

        about()


# ============================================================
# APPLICATION ENTRY
# ============================================================

if not st.session_state.logged_in:

    login_page()

else:

    dashboard()