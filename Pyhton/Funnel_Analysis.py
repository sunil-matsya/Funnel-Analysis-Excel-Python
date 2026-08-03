import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# LOAD DATASET
# ==========================================================

file_path = r"S:\Black Pearl\empty\chat\funnel_events_sample.csv"

df = pd.read_csv(file_path)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Remove duplicate events
df = df.drop_duplicates(subset=["user_id", "step"])

# Funnel order
steps = [
    "visited_site",
    "signup_started",
    "details_filled",
    "email_verified",
    "purchase_completed"
]

df["step"] = pd.Categorical(
    df["step"],
    categories=steps,
    ordered=True
)

# ==========================================================
# FUNNEL COUNTS
# ==========================================================

counts = (
    df.groupby("step", observed=False)["user_id"]
      .nunique()
      .reindex(steps)
)

# ==========================================================
# CONVERSION
# ==========================================================

conversion = [100]

for i in range(1, len(counts)):
    prev = counts.iloc[i-1]
    curr = counts.iloc[i]

    if prev == 0:
        conversion.append(0)
    else:
        conversion.append(curr/prev*100)

# ==========================================================
# DROP OFF
# ==========================================================

drop = [0]

for i in range(1, len(conversion)):
    drop.append(100-conversion[i])

# ==========================================================
# RESULT TABLE
# ==========================================================

result = pd.DataFrame({
    "Users": counts.values,
    "Conversion %": conversion,
    "Drop-off %": drop
}, index=steps)

result = result.round(2)

print("\n")
print("="*70)
print("                 PRODUCT FUNNEL ANALYSIS")
print("="*70)
print(result)

# ==========================================================
# KPI
# ==========================================================

total_users = counts.iloc[0]

completed = counts.iloc[-1]

overall_conversion = completed/total_users*100

largest_stage = result["Drop-off %"].idxmax()

largest_drop = result["Drop-off %"].max()

previous_stage = steps[steps.index(largest_stage)-1]

print("\n")
print("="*70)
print("KPI")
print("="*70)

print("Total Users           :", total_users)
print("Purchases             :", completed)
print("Overall Conversion    : {:.2f}%".format(overall_conversion))
print("Largest Drop-off      :", previous_stage, "->", largest_stage)
print("Drop-off Percentage   : {:.2f}%".format(largest_drop))

# ==========================================================
# TIME TO CONVERT
# ==========================================================

time_table = []

for i in range(len(steps)-1):

    s1 = steps[i]
    s2 = steps[i+1]

    left = df[df["step"]==s1][["user_id","timestamp"]]
    right = df[df["step"]==s2][["user_id","timestamp"]]

    merge = pd.merge(left,right,on="user_id",suffixes=("_1","_2"))

    if len(merge)>0:

        merge["minutes"] = (
            merge["timestamp_2"]-
            merge["timestamp_1"]
        ).dt.total_seconds()/60

        avg = merge["minutes"].mean()

    else:
        avg = 0

    time_table.append({
        "From":s1,
        "To":s2,
        "Average Minutes":round(avg,2)
    })

time_df = pd.DataFrame(time_table)

print("\n")
print("="*70)
print("AVERAGE TIME TO CONVERT")
print("="*70)

print(time_df)

# ==========================================================
# SEGMENT COMPARISON
# ==========================================================

df["number"] = df["user_id"].str.extract("(\d+)").astype(int)

df["segment"] = np.where(
    df["number"]<=100,
    "User <=100",
    "User >100"
)

segment_table = []

for seg in df["segment"].unique():

    temp = df[df["segment"]==seg]

    c = (
        temp.groupby("step", observed=False)["user_id"]
        .nunique()
        .reindex(steps)
    )

    segment_table.append(c.values)

segment_df = pd.DataFrame(
    np.array(segment_table).T,
    index=steps,
    columns=df["segment"].unique()
)

print("\n")
print("="*70)
print("SEGMENT COMPARISON")
print("="*70)

print(segment_df)

print("\nData preparation completed.")
print("Proceed to Visualization...\n")

# ==========================================================
# VISUAL DASHBOARD
# ==========================================================

fig = plt.figure(figsize=(18, 14))

# ----------------------------------------------------------
# 1. FUNNEL CHART (Horizontal Bar)
# ----------------------------------------------------------

ax1 = plt.subplot(3, 2, 1)

funnel_users = result["Users"][::-1]
funnel_steps = result.index[::-1]

ax1.barh(funnel_steps, funnel_users)

ax1.set_title("Funnel Visualization")
ax1.set_xlabel("Users")

for i, value in enumerate(funnel_users):
    ax1.text(value + 2, i, str(value), va='center')

# ----------------------------------------------------------
# 2. USER COUNT BAR CHART
# ----------------------------------------------------------

ax2 = plt.subplot(3, 2, 2)

bars = ax2.bar(result.index, result["Users"])

ax2.set_title("Unique Users at Each Stage")
ax2.set_ylabel("Users")
ax2.tick_params(axis='x', rotation=25)

for bar in bars:
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        h + 2,
        f"{int(h)}",
        ha="center"
    )

# ----------------------------------------------------------
# 3. CONVERSION RATE
# ----------------------------------------------------------

ax3 = plt.subplot(3, 2, 3)

ax3.plot(
    result.index,
    result["Conversion %"],
    marker="o",
    linewidth=3
)

ax3.set_ylim(0, 110)
ax3.grid(True)

ax3.set_title("Conversion Rate")

for i, value in enumerate(result["Conversion %"]):
    ax3.text(i, value + 2, f"{value:.1f}%")

# ----------------------------------------------------------
# 4. DROP-OFF CHART
# ----------------------------------------------------------

ax4 = plt.subplot(3, 2, 4)

colors = []

for value in result["Drop-off %"]:
    if value == largest_drop:
        colors.append("red")
    else:
        colors.append("steelblue")

bars = ax4.bar(
    result.index,
    result["Drop-off %"],
    color=colors
)

ax4.set_title("Drop-off Percentage")
ax4.tick_params(axis='x', rotation=25)

for bar in bars:
    h = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width()/2,
        h + 1,
        f"{h:.1f}%",
        ha="center"
    )

# ----------------------------------------------------------
# 5. TIME TO CONVERT
# ----------------------------------------------------------

ax5 = plt.subplot(3, 2, 5)

labels = []

for i in range(len(time_df)):
    labels.append(
        time_df.loc[i, "From"].replace("_", "\n")
        +
        "\n↓\n"
        +
        time_df.loc[i, "To"].replace("_", "\n")
    )

bars = ax5.bar(
    labels,
    time_df["Average Minutes"]
)

ax5.set_title("Average Time Between Stages")
ax5.set_ylabel("Minutes")

for bar in bars:
    h = bar.get_height()
    ax5.text(
        bar.get_x() + bar.get_width()/2,
        h + 0.5,
        f"{h:.1f}",
        ha="center"
    )

# ----------------------------------------------------------
# 6. SEGMENT COMPARISON
# ----------------------------------------------------------

ax6 = plt.subplot(3, 2, 6)

x = np.arange(len(steps))
width = 0.35

seg1 = segment_df.iloc[:, 0]
seg2 = segment_df.iloc[:, 1]

ax6.bar(
    x - width/2,
    seg1,
    width,
    label=segment_df.columns[0]
)

ax6.bar(
    x + width/2,
    seg2,
    width,
    label=segment_df.columns[1]
)

ax6.set_xticks(x)
ax6.set_xticklabels(
    [s.replace("_", "\n") for s in steps]
)

ax6.legend()

ax6.set_title("Segment Comparison")

plt.suptitle(
    "PRODUCT FUNNEL ANALYSIS DASHBOARD",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()