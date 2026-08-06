import ultraplot as pplt
import polars as pl
import numpy as np
# 直接设置scen变量，而不是通过命令行参数
scen_name = "edrive_singweight"

# 直接设置scen变量，而不是通过命令行参数
scen_name = "edrive_singweight"

# --- 排布：BAU → Circular → Substitution → … → Ideal；颜色红→蓝与 strategy_full_config 一致 ---
strategy_dict = {
    "bau_bau_bau_recycl_nosub": "#d83f3f",   # 1 BAU
    "bau_llt_mir_recych_nosub": "#ff7f2a",   # 2 Circular
    "bau_bau_bau_recycl_sub": "#ff9371",     # 3 Substitution
    "bau_llt_mir_recych_sub": "#ffc46a",     # 4 Circular+Substitution
    "allo_bau_bau_recycl_nosub": "#ffd42a",      # 5 Global Allocation
    "allo_llt_mir_recych_nosub": "#63c6d4",      # 6 Circular+Global Allocation
    "allo_bau_bau_recycl_sub": "#81d5ac",        # 7 Global Allocation+Substitution
    "allo_llt_mir_recych_sub": "#bef28a",        # 8 All-in
    "ideal": "#2988be",                          # 9 Ideal
}


strategy_display_names = {
    "ideal": "Ideal",
    "bau_bau_bau_recycl_nosub": "BAU",
    "bau_llt_mir_recych_nosub": "Circular",
    "bau_bau_bau_recycl_sub": "Substitution",
    "bau_llt_mir_recych_sub": "Circular + Substitution",
    "allo_bau_bau_recycl_nosub": "Global Allocation",
    "allo_llt_mir_recych_nosub": "Circular + Global Allocation",
    "allo_bau_bau_recycl_sub": "Global Allocation + Substitution",
    "allo_llt_mir_recych_sub": "All-in",
}

strategy_full_config = {
    "bau_bau_bau_recycl_nosub": {"color": "#d83f3f", "label": "BAU", "shading": "all"},
    "bau_llt_mir_recych_nosub": {"color": "#ff7f2a", "label": "Circular", "shading": None},
    "bau_bau_bau_recycl_sub": {"color": "#ff9371", "label": "Substitution", "shading": "all"},
    "bau_llt_mir_recych_sub": {"color": "#ffc46a", "label": "Circular + Substitution", "shading": None},
    "allo_bau_bau_recycl_nosub": {"color": "#ffd42a", "label": "Global Allocation", "shading": "None"},
    "allo_llt_mir_recych_nosub": {"color": "#63c6d4", "label": "Circular + Global Allocation", "shading": "None"},
    "allo_bau_bau_recycl_sub": {"color": "#81d5ac", "label": "Global Allocation + Substitution", "shading": "None"},
    "allo_llt_mir_recych_sub": {"color": "#bef28a", "label": "All-in", "shading": "None"},
    "ideal": {"color": "#2988be", "label": "Ideal", "shading": "all"},
}







# 确保 cf_colors 足够多
cf_colors = ["#0072B2", "#56B4E9", "#F5C710", "#E6E6FA", "#009E73", "#D55E00", "#F0E442"] # Added a couple more colors

print("Loading historical data...")
hist = pl.read_parquet(
    f"../../data/data_raw/miracle_oscar/{scen_name}/dtg_weighted_hist_2000mc.parquet"
).sort("year")

print("Loading scenario data (scen)...")
scen = (
    pl.scan_parquet(
        f"../../data/data_raw/miracle_oscar/{scen_name}/dtg_weighted_scen_2000mc.parquet"
    )
    .with_columns(
        pl.col("scen")
        .str.splitn("|", 2)
        .struct.rename_fields(["model", "scenario"])
    )
    .unnest("scen")
    .with_columns(
        strategy=pl.lit('ideal') # scen 数据视为 'ideal' 策略
    )
    .select(pl.col('model', 'scenario', 'category', 'strategy', 'measure', 'year', 'value'))
    .sort("year")
)

print("Loading counterfactual data (cf)...")
cf = (
    pl.scan_parquet(
        f"../../data/data_raw/miracle_oscar/{scen_name}/dtg_weighted_cf_2000mc_category*_scen_all.parquet"
    )
    .with_columns(
        pl.col("scen")
        .str.splitn("|", 3)
        .struct.rename_fields(["model", "scenario", "strategy"])
    )
    .unnest("scen")
    .filter(pl.col("strategy").is_in(strategy_dict.keys()))
    .select(pl.col('model', 'scenario', 'category', 'strategy', 'measure', 'year', 'value'))
    .sort("year")
)

print("Combining scen and cf data...")
# Collect early if memory allows and data is not too large.
# If data is very large, consider processing in lazy mode where possible.
scen_stats = pl.concat([scen.collect(), cf.collect()])


# 获取必要的变量
hist_year = hist["year"].unique().sort()
hist_mean = hist.filter(pl.col("measure") == "mean")["value"]
# hist_std = hist.filter(pl.col("measure") == "std")["value"] # hist_std was loaded but not used

future_year = scen_stats['year'].unique().sort()
# Use scen_stats to get categories relevant to the plot
categories_series = scen_stats["category"].unique().sort() # Get unique categories as Series

# 创建图表
pplt.rc["grid"] = False
# Adjusted figure width slightly for better layout
fig = pplt.figure(refwidth=4, facecolor=(0, 0, 0, 0)) # <<< Added facecolor=(0, 0, 0, 0)
axes = fig.subplots(nrows=1, ncols=1)

# Determine the category to plot (C1 preferred, fallback to second if C1 not found)
target_category = 'C2'
# Check if the target category exists in the unique categories series
if target_category in categories_series.to_list(): # Use to_list() for a simple 'in' check
    cate = target_category
    print(f"Found and focusing on category: '{cate}'")
elif categories_series.len() > 1:
    # If the target category is not found, use the second category as a fallback
    print(f"Category '{target_category}' not found. Using the second category as fallback.")
    # Safely get the second item (index 1) from the Series using .item()
    cate = categories_series.item(1)
    print(f"Fallback category: '{cate}'")
elif categories_series.len() == 1:
     # If only one category exists, use that one
     cate = categories_series.item(0)
     print(f"Only one category found. Using: '{cate}'")
else:
    # Handle the case where no categories are found
    raise ValueError("No categories found in data.")


# Calculate pre-industrial base temperature (1850-1900 mean)
hist_year_industrial_age_base_mean = (
    hist.filter(pl.col("year").is_between(1850, 1900))
    .filter(pl.col("measure") == "mean")
    .select(pl.mean(["value"]))
    .to_numpy()
    .ravel()
)
base_temp = hist_year_industrial_age_base_mean.item()

print("Plotting historical data")
# Plot historical data, shifted relative to pre-industrial base
# Only plot historical data up to the start of the future years for clarity on the main axes
hist_plot_years = hist_year.filter(hist_year <= future_year.min())
hist_plot_mean = hist_mean.filter(hist_year <= future_year.min())
axes.plot(hist_plot_years, hist_plot_mean - base_temp, color="k", lw=1.5, label="Historical")

px = axes.panel('r', width='8em')

span_xmin = 2020
span_xmax = 2100

axes.hlines(1.5, span_xmin, span_xmax, color="gray", linestyle="--", lw=0.75, label="1.5°C Threshold", zorder=1)
axes.hlines(2.0, span_xmin, span_xmax, color="gray", linestyle="--", lw=0.75, label="2.0°C Threshold", zorder=1)

hs = [] # Handles for the legend
scen_median_data = {} # Dictionary to store median data for divergence calculation

for strategy_internal_name in strategy_full_config.keys():
    config = strategy_full_config[strategy_internal_name]
    color = config["color"]
    shading_type = config.get("shading", "all")

    print(f"Processing data for strategy: {strategy_internal_name} (Shading: {shading_type})")
    # Filter data using the internal strategy name
    strategy_category_data = (
        scen_stats.filter(pl.col("category")==cate)
        .filter(pl.col("measure") == "mean")
        .filter(pl.col("strategy") == strategy_internal_name)
    )

    if strategy_category_data.is_empty():
        print(f"No data found for strategy '{strategy_internal_name}' in category '{cate}'. Skipping plot.")
        continue

    # Pivot data
    pivot_result = strategy_category_data.pivot(
         index=['model', 'scenario'],
         on='year',
         values='value',
    )

    # Offset values
    scen_stats_offset = (
        pivot_result.with_columns(
            [pl.col(col) - base_temp for col in pivot_result.columns if col not in ['model', 'scenario']]
        )
    )

    # Unpivot
    year_value_cols = [col for col in scen_stats_offset.columns if col not in ['model', 'scenario']]

    if not year_value_cols:
        print(f"No year/value columns found after pivot for strategy '{strategy_internal_name}'. Skipping plot.")
        continue

    scen_stats_unpivoted = scen_stats_offset.unpivot(
         index=["model", "scenario"],
         on=year_value_cols,
         variable_name="year_str",
         value_name="value_offset"
    ).with_columns(
        pl.col("year_str").cast(pl.Int64).alias("year")
    ).select(pl.col('model', 'scenario', 'year', 'value_offset')).sort("year")


    # Calculate stats per year across model/scenario runs
    if scen_stats_unpivoted.is_empty():
         print(f"No data after unpivoting for strategy '{strategy_internal_name}'. Skipping plot.")
         continue

    scen_median_cate = scen_stats_unpivoted.group_by('year').agg(pl.median('value_offset').alias('value_median')).sort('year')
    scen_median_data[strategy_internal_name] = scen_median_cate.clone()

    print(f"Plotting {strategy_internal_name} in {cate}")

    plot_data = scen_median_cate.filter(pl.col("year").is_in(future_year.to_list()))
    if not plot_data.is_empty():
        m = axes.plot(
            plot_data['year'],
            plot_data['value_median'],
            lw=1.5,
            label=strategy_internal_name,
            color=color,
            zorder=4,
        )
        hs.extend(m)
    else:
        print(f"No plot data generated for strategy '{strategy_internal_name}' in category '{cate}' after joins and filtering for future years.")

# # --- Calculate and Plot 1.5°C Crossing Years ---
# print("Calculating and plotting 1.5°C crossing years...")
# crossing_year_data = {}
# # We only look for crossings AFTER the historical period, typically from 2000 or 2020 onwards
# crossing_start_year = 2000
# crossing_marker_handles = [] # List to store handles for crossing markers

# for strategy, median_df in scen_median_data.items():
#     # Ensure median_df is not empty and covers relevant years
#     if not median_df.is_empty():
#         # Find the first year where median >= 1.5, starting from crossing_start_year
#         crossing_point = median_df.filter(pl.col("year") >= crossing_start_year).filter(pl.col("value_median") >= 1.5).sort("year").head(1)
#         if not crossing_point.is_empty():
#             year = crossing_point['year'][0]
#             value_at_year = crossing_point['value_median'][0] # Get the exact median value at the crossing year
#             crossing_year_data[strategy] = year
#             print(f"Strategy '{strategy}' crosses 1.5°C in {year}")
#             # Plot a marker at the crossing year (on the strategy's median line)
#             marker_handle = axes.plot(year, value_at_year, marker='o', markersize=5, color=strategy_dict[strategy], zorder=10)
#             crossing_marker_handles.extend(marker_handle) # Add handle
#             # Add text label slightly above/right of the marker (Optional)
#             # axes.text(year + 1, value_at_year + 0.05, str(year), ha="left", va="bottom", color=strategy_dict[strategy], fontsize=7)

# --- Calculate and Plot Ideal Peak Warming ---
print("Calculating and plotting 'ideal' peak warming year...")
ideal_peak_marker_handle = None # Initialize handle

if 'ideal' in scen_median_data:
    ideal_median_df = scen_median_data['ideal']

    # Filter for future years (e.g., 2000 onwards) to find the peak
    # We take max from the entire available future period
    ideal_future_df = ideal_median_df.filter(pl.col('year') >= future_year.min()) # Start from the first future year

    if not ideal_future_df.is_empty():
        # Find the row with the maximum median value
        peak_row = ideal_future_df.sort('value_median', descending=True).head(1)

        if not peak_row.is_empty():
            peak_year = peak_row['year'][0]
            peak_value = peak_row['value_median'][0]
            print(f"'ideal' peak warming occurs in {peak_year} at {peak_value:.2f}°C")

            # Plot the peak marker
            # Use a unique marker like a star '*'
            # ideal_peak_marker_handle = axes.plot(
            #     peak_year, peak_value,
            #     marker='*',          # Star marker
            #     markersize=8,        # Larger size to be visible
            #     color=strategy_dict['ideal'], # Use ideal color
            #     markeredgecolor='k', # Black edge for visibility
            #     markeredgewidth=0.7,
            #     zorder=11,           # Make sure it's on top
            #     label="ideal Peak Warming" # Label for the legend
            # )
             # plot returns a list of handles, we need the first one
            if ideal_peak_marker_handle:
                 ideal_peak_marker_handle = ideal_peak_marker_handle[0]
        else:
             print("Could not find peak in 'ideal' future data.")
    else:
        print("'ideal' future data is empty, cannot find peak.")
else:
    print("'ideal' strategy data not available to find peak warming.")

print("Calculating and plotting 'ideal' peak warming level as a horizontal line...")
ideal_peak_line_handle = None # Initialize handle for the line
ideal_peak_value = None # Initialize peak value
ideal_peak_year = None # Initialize peak year (useful for the next section)


if 'ideal' in scen_median_data:
    ideal_median_df = scen_median_data['ideal']

    # Filter for future years (e.g., from the minimum future year in the data) to find the peak
    ideal_future_df = ideal_median_df.filter(pl.col('year') >= future_year.min()) # Start from the first future year

    if not ideal_future_df.is_empty():
        # Find the row with the maximum median value
        peak_row = ideal_future_df.sort('value_median', descending=True).head(1)

        if not peak_row.is_empty():
            peak_year = peak_row['year'][0]
            peak_value = peak_row['value_median'][0]
            print(f"'ideal' peak warming occurs in {peak_year} at {peak_value:.2f}°C")

            # Store these values as they are needed in the next code block
            ideal_peak_year = peak_year # <<< Store Ideal Peak Year
            ideal_peak_value = peak_value # <<< Store Ideal Peak Value

            # Plot the peak warming level as a horizontal dashed line (贯穿图表)
            # Use the same span as the colored zones and other hlines
            # span_xmin and span_xmax should be defined earlier for colored zones
            ideal_peak_line_handle = axes.hlines( # <<< This plots the full horizontal line
                peak_value,
                span_xmin, # Use the defined span for colored zones/other hlines
                span_xmax,
                color=strategy_full_config['ideal']["color"], # Use ideal color from full config
                linestyle='--', # Dashed line, like 1.5C and 2.0C
                lw=0.75, # Line width
                label="ideal Peak Temp.", # Label for the legend
                zorder=1 # Same zorder as 1.5/2.0 hlines
            ) # <<< REMOVE [0] here if it was present before

        else:
             print("Could not find peak in 'ideal' future data.")
    else:
        print("'ideal' future data is empty, cannot find peak.")
else:
    print("'ideal' strategy data not available to find peak warming.")

# --- Boxplot Section ---
print("Plotting 2100 boxplots in right panel")

# Collect all 2100 offset values for each strategy
boxplot_2100_data = {}
# Re-filter from scen_stats collected DataFrame for 2100 values
for strategy, color in strategy_dict.items():
    strategy_2100_values_df = (
        scen_stats.filter(pl.col("category")==cate)
        .filter(pl.col("measure") == "mean")
        .filter(pl.col("strategy") == strategy)
        .filter(pl.col("year") == 2100)
        .select((pl.col('value') - base_temp).alias('value_offset_2100'))
    )

    if not strategy_2100_values_df.is_empty():
        boxplot_2100_data[strategy] = strategy_2100_values_df['value_offset_2100'].to_numpy().ravel()
    else:
         print(f"No data for 2100 boxplot for strategy '{strategy}'. Skipping.")


desired_boxplot_order_keys = list(strategy_full_config.keys()) # Or your custom order list

# Initialize empty lists for boxplot data, which will be populated in the desired order
boxplot_positions = []
boxplot_stats = []
strategy_labels_boxplot = [] # Store internal keys for x-axis labels
strategy_colors_boxplot = [] # Colors for boxplot bars

spacing = 1.0 # Adjust spacing between boxes

# Loop through the desired order of strategy keys
for i, key in enumerate(desired_boxplot_order_keys):
    # Check if you have raw data for this strategy key in boxplot_2100_data
    if key in boxplot_2100_data:
        values_2100 = boxplot_2100_data[key] # Get the raw values
        if len(values_2100) > 0:
            # Calculate quantiles for the collected values
            p05, p25, p50, p75, p95 = np.percentile(values_2100, [5, 25, 50, 75, 95])

            # Append the calculated data and attributes IN THE CURRENT ORDER
            boxplot_stats.append({
                'median': p50,
                'p05': p05,
                'p25': p25,
                'p75': p75,
                'p95': p95
            })
            boxplot_positions.append(i * spacing) # Position based on the desired order (index i)
            strategy_labels_boxplot.append(key) # Store the internal key
            # Get color from the config dictionary using the key
            strategy_colors_boxplot.append(strategy_full_config.get(key, {'color': 'gray'})['color']) # Use .get for safety


        else:
            print(f"Warning: No values found for 2100 boxplot for strategy '{key}'. Skipping.")
    else:
         print(f"No raw data found for 2100 boxplot for strategy '{key}'. Skipping.")


print("Calculating temperature differences using Method 2 (median of raw differences)...")

# Calculate raw differences (strategy - ideal) for each model/scenario run, then take median
strategy_diff_medians = {}

# Get ideal data for comparison
ideal_2100_data = boxplot_2100_data.get("ideal", np.array([]))

if len(ideal_2100_data) > 0:
    # For each strategy, calculate median of (strategy_value - ideal_value) for each model run
    for strategy, strategy_2100_data in boxplot_2100_data.items():
        if strategy != "ideal" and len(strategy_2100_data) > 0:
            # Calculate differences for each model/scenario run
            # We need to match model/scenario runs between strategies and ideal

            # Get raw data from scen_stats for both strategy and ideal
            strategy_raw = (
                scen_stats.filter(pl.col("category")==cate)
                .filter(pl.col("measure") == "mean")
                .filter(pl.col("strategy") == strategy)
                .filter(pl.col("year") == 2100)
                .select(['model', 'scenario', (pl.col('value') - base_temp).alias('strategy_temp')])
            )

            ideal_raw = (
                scen_stats.filter(pl.col("category")==cate)
                .filter(pl.col("measure") == "mean")
                .filter(pl.col("strategy") == "ideal")
                .filter(pl.col("year") == 2100)
                .select(['model', 'scenario', (pl.col('value') - base_temp).alias('ideal_temp')])
            )

            # Join and calculate differences for matched model/scenario runs
            if not strategy_raw.is_empty() and not ideal_raw.is_empty():
                differences = (
                    strategy_raw.join(ideal_raw, on=['model', 'scenario'], how='inner')
                    .with_columns((pl.col('strategy_temp') - pl.col('ideal_temp')).alias('temp_diff'))
                )

                if not differences.is_empty():
                    # Calculate median of raw differences (Method 2)
                    diff_median = differences['temp_diff'].median()
                    strategy_diff_medians[strategy] = diff_median
                    print(f"Strategy {strategy}: median difference = {diff_median:.3f}°C")

bar_width = 1
for j, stats in enumerate(boxplot_stats):
    pos = boxplot_positions[j]
    color = strategy_colors_boxplot[j]
    px.bar(pos, stats['p75'] - stats['p25'], bottom=stats['p25'], width=bar_width, color=color, alpha=0.75, zorder=5)
    px.hlines(stats['median'], pos - bar_width / 2, pos + bar_width / 2, color='k', lw=1.5, zorder=6)
    px.bar(pos, stats['p95'] - stats['p05'], bottom=stats['p05'], width=bar_width * 0.5, color=color, alpha=0.35, zorder=4)

zero_tolerance = 0.01
for j, stats in enumerate(boxplot_stats):
    strategy = strategy_labels_boxplot[j]
    pos = boxplot_positions[j]

    if strategy != "ideal" and strategy in strategy_diff_medians:
        diff = strategy_diff_medians[strategy]
        if abs(diff) > zero_tolerance:
            diff_text = f"{diff:.2f}°C" if diff < 0 else f"+{diff:.2f}°C"
            text_va = "bottom" if diff >= 0 else "top"
            text_y = stats['median'] + 0.08 if diff >= 0 else stats['median'] - 0.08
            px.text(pos + bar_width / 2 + 0.05, text_y, diff_text, ha="left", va=text_va, color='k', fontsize=8, weight='bold', zorder=10)

px.number = 2
px.format(
    abc=True,
    abcloc="ul",
    xlim=(-bar_width / 2, boxplot_positions[-1] + bar_width / 2 + spacing / 2) if boxplot_positions else (0, 1),
    tickminor=False,
    ytickloc='none',
    ylim=axes.get_ylim(),
)


axes.format(xlabel="Year")

axes.format(
    abc=True,
    abcloc="ul",
    tickminor=False,
    xticks=[2025, 2050, 2100],
    yticks=[1, 1.5, 2, 2.5],
    xlim=(2025, 2100),
    ylim=(1, 2.5),
    ylabel="Temperature change relative to pre-industrial (°C)",
)

# Add a single legend for the main lines
# Combine line and threshold labels
# ... (箱型图部分结束) ...

# Add a single legend for the main lines and collected handles
# hs contains handles for the main strategy median lines (labels are internal names)
# axes.get_lines()[1:] should get handles for hlines, other markers/lines (labels are set explicitly or default)
all_handles = hs + axes.get_lines()[1:]
all_labels_raw = [h.get_label() for h in all_handles] # Get raw labels (internal names or explicit)

# --- Replace internal strategy names with display names in the labels ---
all_labels_display = []
for raw_label in all_labels_raw:
    # If the raw label is an internal strategy name in our display map, use the display name
    if raw_label in strategy_display_names:
        all_labels_display.append(strategy_display_names[raw_label])
    else:
        # Otherwise, keep the original label (e.g., "Historical", "1.5°C Threshold", "ideal Peak Temp.")
        all_labels_display.append(raw_label)

# Filter for unique handles and labels before creating legend (optional but good practice)
unique_labels = []
unique_handles = []
for handle, label in zip(all_handles, all_labels_display):
    # We can use the display label for uniqueness check
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)
    # else: # If label is a duplicate, check if the handle is the same type or needed
         # You might need more sophisticated logic here if different plot calls
         # produce handles with the same default label unintentionally.
         # For now, simple uniqueness by display label is usually sufficient.


fig.legend(unique_handles, unique_labels, ncols=3, center=True, frame=False, loc="bottom", order='F') # Use unique handles/labels and order='F'

# ... (后续标题、布局、保存代码) ...

# 保存图片
fig.save(f"../../testpic/temperature_hist_scen_{scen_name}_C2_cross_div.pdf")

print("Plot generated and saved.")
