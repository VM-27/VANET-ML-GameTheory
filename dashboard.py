import streamlit as st
import matplotlib.pyplot as plt
import subprocess
import time
import re
from pathlib import Path

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="🚗 VANET AI Dashboard", layout="wide")

# --------------------------
# TITLE
# --------------------------
st.markdown("<h1 style='text-align:center; color:#00FFFF;'>🚗 VANET AI DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>ML + Game Theory + Network Simulation</h4>", unsafe_allow_html=True)

# --------------------------
# PATHS
# --------------------------
NS3_DIR = Path.home() / "ns-3-dev"

# --------------------------
# PARSE FUNCTION
# --------------------------
def parse_metrics():
    try:
        result = subprocess.check_output(
            ["python3", "parse_flowmon_cluster.py", "flowmon_cluster_traffic.xml"],
            cwd=NS3_DIR,
            text=True
        )
        pdr = float(re.search(r"PDR \(%\): ([0-9.]+)", result).group(1))
        delay = float(re.search(r"Avg Delay \(s\): ([0-9.]+)", result).group(1))
        thr = float(re.search(r"Throughput \(Mbps\): ([0-9.]+)", result).group(1))
        return pdr, delay, thr
    except:
        return None, None, None

# --------------------------
# TABS
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Simulation",
    "📡 Routing Comparison",
    "🤖 ML vs Game Theory",
    "📊 Live Analytics"
])

# =========================================================
# 🚀 TAB 1: SIMULATION CONTROL
# =========================================================
with tab1:
    st.subheader("Run VANET Simulation")

    if st.button("▶ Run Simulation"):
        with st.spinner("Running ns-3..."):
            subprocess.run(
                "./ns3 run 'scratch/vanet_cluster_traffic --nNodes=50 --simTime=100 --membersSend=true --chRate=20kbps --memberRate=10kbps --pktSize=256'",
                shell=True,
                cwd=NS3_DIR
            )
        st.success("Simulation Completed!")

    st.info("Use NetAnim separately for visualization.")

# =========================================================
# 📡 TAB 2: ROUTING COMPARISON
# =========================================================
with tab2:
    st.subheader("AODV vs OLSR")

    protocols = ["AODV", "OLSR"]
    PDR = [23, 68]
    Delay = [0.02, 0.008]

    col1, col2 = st.columns(2)

    fig1, ax1 = plt.subplots()
    ax1.bar(protocols, PDR, color=["red", "green"])
    ax1.set_title("PDR Comparison")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(protocols, Delay, color=["orange", "blue"])
    ax2.set_title("Delay Comparison")
    col2.pyplot(fig2)

    st.success("✔ OLSR significantly outperforms AODV in VANET")

# =========================================================
# 🤖 TAB 3: ML vs GAME THEORY
# =========================================================
with tab3:
    st.subheader("Static ML vs Dynamic GT")

    methods = ["Static ML", "Dynamic GT"]
    PDR_vals = [79.37, 77.88]
    Delay_vals = [0.0069, 0.0163]
    Thr_vals = [0.006, 0.002]

    # KPI CARDS
    col1, col2, col3 = st.columns(3)
    col1.metric("Best PDR", f"{max(PDR_vals)}%")
    col2.metric("Lowest Delay", f"{min(Delay_vals)} s")
    col3.metric("Best Throughput", f"{max(Thr_vals)} Mbps")

    # GRAPHS
    col1, col2 = st.columns(2)

    fig3, ax3 = plt.subplots()
    ax3.bar(methods, PDR_vals, color=["cyan", "purple"])
    ax3.set_title("PDR")
    col1.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    ax4.bar(methods, Delay_vals)
    ax4.set_title("Delay")
    col2.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    ax5.bar(methods, Thr_vals)
    ax5.set_title("Throughput")
    st.pyplot(fig5)

    best_method = methods[PDR_vals.index(max(PDR_vals))]
    st.success(f"🏆 Best Method: {best_method}")

# =========================================================
# 📊 TAB 4: LIVE ANALYTICS
# =========================================================
with tab4:
    st.subheader("Live Metrics")

    pdr, delay, thr = parse_metrics()

    col1, col2, col3 = st.columns(3)
    col1.metric("PDR (%)", pdr if pdr else "--")
    col2.metric("Delay (s)", delay if delay else "--")
    col3.metric("Throughput (Mbps)", thr if thr else "--")

    if st.checkbox("🔄 Enable Live Monitoring"):
        placeholder = st.empty()
        for _ in range(10):
            pdr, delay, thr = parse_metrics()
            if pdr:
                fig, ax = plt.subplots()
                ax.bar(["PDR", "Delay", "Throughput"], [pdr, delay, thr])
                placeholder.pyplot(fig)
            time.sleep(2)

    # AUTO ANALYSIS
    if pdr:
        st.info(f"""
        📌 Analysis:
        - Current PDR = {pdr}%
        - Delay = {delay}s
        - Throughput = {thr} Mbps

        ✔ Cluster-based routing improves communication.
        ✔ Game theory introduces adaptability.
        ⚠ Further tuning can reduce delay.
        """)
    else:
        st.warning("Run simulation first to see analysis.")

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.markdown("🚀 Developed as part of VANET ML + Game Theory Project")
