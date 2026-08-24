import numpy as np
import pandas as pd
 
RAW_DIR = "/home/claude/coldstart_project/azure_raw"
DAY = "01"  # using day 1 of the 14-day trace; can extend to more days later
 
# --- Load the three real trace files for this day ---
durations = pd.read_csv(f"{RAW_DIR}/function_durations_percentiles.anon.d{DAY}.csv")
memory = pd.read_csv(f"{RAW_DIR}/app_memory_percentiles.anon.d{DAY}.csv")
invocations = pd.read_csv(f"{RAW_DIR}/invocations_per_function_md.anon.d{DAY}.csv")
 
print(f"Raw durations rows: {len(durations)}")
print(f"Raw memory rows (per app): {len(memory)}")
print(f"Raw invocations rows: {len(invocations)}")
 
# --- Invocations: sum the 1440 per-minute columns into a daily total, keep trigger ---
minute_cols = [c for c in invocations.columns if c.isdigit()]
invocations["invocations_per_day"] = invocations[minute_cols].sum(axis=1)
inv_slim = invocations[["HashOwner", "HashApp", "HashFunction", "Trigger", "invocations_per_day"]]
 
# --- Durations: use the Average execution time column ---
dur_slim = durations[["HashOwner", "HashApp", "HashFunction", "Average", "Count"]].rename(
    columns={"Average": "avg_duration_ms"}
)
 
# --- Memory: this is at the APPLICATION level (real dataset constraint,
# same as noted in the paper), so we join it onto every function in that app ---
mem_slim = memory[["HashOwner", "HashApp", "AverageAllocatedMb"]].rename(
    columns={"AverageAllocatedMb": "memory_mb"}
)
 
# --- Merge everything on function/app identity ---
merged = inv_slim.merge(dur_slim, on=["HashOwner", "HashApp", "HashFunction"], how="inner")
merged = merged.merge(mem_slim, on=["HashOwner", "HashApp"], how="inner")
 
print(f"\nAfter merging all three real files: {len(merged)} functions with complete data")
 
# --- Build task_id and clean up trigger type naming to match our pipeline ---
merged["task_id"] = merged["HashFunction"].str[:12]
merged["trigger_type"] = merged["Trigger"].str.capitalize()
 
# --- has_history: real functions with very few invocations behave like
# zero-history tasks (barely any signal to learn from); functions with many
# invocations have rich history. We use invocation count as the real-data
# proxy for "history", exactly the situation Shahrad et al. describe where
# most functions are invoked rarely. ---
history_threshold = merged["invocations_per_day"].quantile(0.3)
merged["has_history"] = merged["invocations_per_day"] > history_threshold
 
final = merged[["task_id", "trigger_type", "avg_duration_ms", "memory_mb",
                "invocations_per_day", "has_history"]].copy()
 
# Drop any degenerate rows (zero/negative duration or memory, which exist
# in small numbers in the real trace due to logging edge cases noted in
# the dataset's own documentation)
final = final[(final["avg_duration_ms"] > 0) & (final["memory_mb"] > 0)]
 
print(f"\nFinal clean real-data task set: {len(final)} tasks")
print(f"Zero-history tasks: {(~final['has_history']).sum()}")
print(f"Trigger type distribution:\n{final['trigger_type'].value_counts()}")
 
final.to_csv("/Users/srira/OneDrive/Desktop/OSProject/tasks_real.csv", index=False)
print("\nSaved -> tasks_real.csv (REAL Azure Functions data)")