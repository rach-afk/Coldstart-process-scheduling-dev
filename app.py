"""
Streamlit demo for: A Lightweight Few-Shot Approach to Burst-Time
Classification for Zero-History Processes.

Run with: streamlit run app.py
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
import simpy

st.set_page_config(page_title="Zero-History Burst-Time Classification", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Pipeline Controls")
n_tasks = st.sidebar.slider("Number of tasks to sample", 200, 2000, 600, step=50)
k_shot = st.sidebar.slider("K-shot: examples per family", 2, 10, 5)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: official Azure Functions 2019 trace (Shahrad et al., "
    "USENIX ATC 2020), downloaded from Azure/AzurePublicDataset on GitHub. "
    "44,249 real functions from Microsoft's production servers."
)

st.title("A Lightweight Few-Shot Approach to Burst-Time Classification for Zero-History Processes")


# ---------------------------------------------------------------------------
# Stage 1: Data generation
# ---------------------------------------------------------------------------
@st.cache_data
def load_real_tasks(n, seed):
    real_df = pd.read_csv("/Users/srira/OneDrive/Desktop/OSProject/tasks_real.csv")
    return real_df.sample(n=min(n, len(real_df)), random_state=seed).reset_index(drop=True)

df = load_real_tasks(n_tasks, seed)
st.success(f"Using {len(df)} REAL functions sampled from the official Azure Functions 2019 trace (Shahrad et al., USENIX ATC 2020).")

st.header("1. Task Dataset")
st.write(
    "Structured like the real Azure Functions trace: trigger type, average "
    "duration, memory allocation, and invocation frequency."
)
c1, c2, c3 = st.columns(3)
c1.metric("Total tasks", len(df))
c2.metric("Zero-history tasks", int((~df["has_history"]).sum()))
c3.metric("Tasks with history", int(df["has_history"].sum()))
st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# Stage 2: Clustering into task families
# ---------------------------------------------------------------------------
st.header("2. Task Family Derivation (K-Means on proxy features)")
st.write(
    "The dataset has no direct CPU/Memory/IO label, so families are derived "
    "by clustering on duration, memory, invocation frequency, and trigger type."
)

features = df[["avg_duration_ms", "memory_mb", "invocations_per_day"]].copy()
trigger_dummies = pd.get_dummies(df["trigger_type"], prefix="trig")
features = pd.concat([features, trigger_dummies], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

summary = df.groupby("cluster")[["avg_duration_ms", "memory_mb", "invocations_per_day"]].mean()
norm = (summary - summary.mean()) / summary.std()
trait_to_label = {"avg_duration_ms": "CPU-heavy", "memory_mb": "Memory-heavy", "invocations_per_day": "IO-heavy"}
scored = [(norm.loc[c, t], c, l) for c in norm.index for t, l in trait_to_label.items()]
scored.sort(reverse=True)
label_map, used = {}, set()
for score, c, l in scored:
    if c in label_map or l in used:
        continue
    label_map[c] = l
    used.add(l)
df["family"] = df["cluster"].map(label_map)

colA, colB = st.columns([1, 1])
with colA:
    st.write("Cluster centers (mean feature values):")
    st.dataframe(summary.style.format("{:.1f}"), use_container_width=True)
with colB:
    st.write("Family distribution:")
    fig, ax = plt.subplots(figsize=(4, 3))
    df["family"].value_counts().plot(kind="bar", ax=ax, color=["#4A6B8B", "#4A8B5C", "#8B4A4A"])
    ax.set_ylabel("Task count")
    plt.xticks(rotation=20)
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Stage 3: Few-shot classifier
# ---------------------------------------------------------------------------
st.header("3. Few-Shot Classifier for Zero-History Tasks")
st.write(
    f"Trained on only **{k_shot} examples per family**, classifies zero-history "
    "tasks using nearest-centroid distance — no iterative training phase."
)

FEATURES = ["avg_duration_ms", "memory_mb", "invocations_per_day"]
history_df = df[df["has_history"]].copy()
zero_history_df = df[~df["has_history"]].copy()

reference_rows = []
for family in history_df["family"].unique():
    fam_pool = history_df[history_df["family"] == family]
    reference_rows.append(fam_pool.sample(n=min(k_shot, len(fam_pool)), random_state=seed))
reference_df = pd.concat(reference_rows)

fs_scaler = StandardScaler()
X_ref = fs_scaler.fit_transform(reference_df[FEATURES])
y_ref = reference_df["family"]
clf = NearestCentroid()
clf.fit(X_ref, y_ref)

X_new = fs_scaler.transform(zero_history_df[FEATURES])
start = time.perf_counter()
predictions = clf.predict(X_new)
elapsed = time.perf_counter() - start
zero_history_df["predicted_family"] = predictions

accuracy = (zero_history_df["predicted_family"] == zero_history_df["family"]).mean()
per_task_latency_us = (elapsed / max(len(zero_history_df), 1)) * 1e6

m1, m2, m3 = st.columns(3)
m1.metric("Reference examples used", f"{len(reference_df)} ({k_shot}/family)")
m2.metric("Classification accuracy", f"{accuracy:.1%}")
m3.metric("Avg inference time / task", f"{per_task_latency_us:.2f} µs")

# ---------------------------------------------------------------------------
# Stage 4: Scheduling simulation
# ---------------------------------------------------------------------------
st.header("4. Scheduling Simulation (SimPy)")
st.write("Comparing FCFS, naive Exponential-Averaging SJF, and our few-shot-enhanced approach.")

rng2 = np.random.default_rng(seed + 1)
df["true_burst_ms"] = df["avg_duration_ms"] * rng2.uniform(0.8, 1.2, len(df))
family_avg_duration = df.groupby("family")["avg_duration_ms"].mean().to_dict()

CPU_CAPACITY = 2

def run_scheduler(policy_name, task_order_df, env, cpu):
    log = []
    def task_process(task):
        arrival = env.now
        with cpu.request() as req:
            yield req
            wait_time = env.now - arrival
            yield env.timeout(task["true_burst_ms"])
            log.append({"task_id": task["task_id"], "policy": policy_name,
                        "wait_ms": wait_time, "burst_ms": task["true_burst_ms"],
                        "turnaround_ms": wait_time + task["true_burst_ms"]})
    procs = []
    for _, task in task_order_df.iterrows():
        procs.append(env.process(task_process(task)))
        yield env.timeout(0)
    yield simpy.events.AllOf(env, procs)
    return log

def simulate(policy_name, ordered_df):
    env = simpy.Environment()
    cpu = simpy.Resource(env, capacity=CPU_CAPACITY)
    logs = []
    def collector():
        result = yield env.process(run_scheduler(policy_name, ordered_df, env, cpu))
        logs.extend(result)
    env.process(collector())
    env.run()
    return pd.DataFrame(logs)

fcfs_order = df.copy()

dataset_avg = df["avg_duration_ms"].mean()
ea_sjf_order = df.copy()
ea_sjf_order["est_burst_ms"] = np.where(ea_sjf_order["has_history"], ea_sjf_order["avg_duration_ms"], dataset_avg)
ea_sjf_order = ea_sjf_order.sort_values("est_burst_ms")

zero_hist_est = zero_history_df.set_index("task_id")["predicted_family"].map(family_avg_duration)
fewshot_order = df.copy()
fewshot_order["est_burst_ms"] = fewshot_order["avg_duration_ms"]
mask = ~fewshot_order["has_history"]
fewshot_order.loc[mask, "est_burst_ms"] = fewshot_order.loc[mask, "task_id"].map(zero_hist_est)
fewshot_order = fewshot_order.sort_values("est_burst_ms")

with st.spinner("Running simulation..."):
    results = {}
    for name, order in [("FCFS", fcfs_order),
                         ("EA-SJF (naive cold tasks)", ea_sjf_order),
                         ("EA-SJF + Few-Shot (ours)", fewshot_order)]:
        results[name] = simulate(name, order)

summary_tbl = pd.DataFrame({
    name: {"Avg Waiting (ms)": r["wait_ms"].mean(), "Avg Turnaround (ms)": r["turnaround_ms"].mean()}
    for name, r in results.items()
}).T

col1, col2 = st.columns([1, 1.3])
with col1:
    st.dataframe(summary_tbl.style.format("{:.1f}"), use_container_width=True)
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    colors = ["#8B4A4A", "#4A6B8B", "#4A8B5C"]
    ax2.bar(summary_tbl.index, summary_tbl["Avg Waiting (ms)"], color=colors)
    ax2.set_ylabel("Avg Waiting Time (ms)")
    plt.xticks(rotation=15, ha="right")
    st.pyplot(fig2)

improvement = (summary_tbl.loc["EA-SJF (naive cold tasks)", "Avg Waiting (ms)"] -
               summary_tbl.loc["EA-SJF + Few-Shot (ours)", "Avg Waiting (ms)"])
st.success(
    f"Our approach reduces average waiting time by **{improvement:.0f} ms** "
    f"compared to naive handling of zero-history tasks, while adding only "
    f"**{per_task_latency_us:.2f} microseconds** of classification overhead per task."
)

