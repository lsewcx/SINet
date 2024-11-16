# 定义两组指标
metrics_model_1 = {
    "MAE": 0.1711,
    "E-measure": 0.2499,
    "F-measure": 0.5660,
    "F_beta": 0.3478,
    "S_object": 0.9029,
    "S_region": 0.3398,
    "Structure_measure": 0.6214
}

metrics_model_2 = {
    "MAE": 0.0429,
    "E-measure": 0.2498,
    "F-measure": 0.6863,
    "F_beta": 0.7183,
    "S_object": 0.9303,
    "S_region": 0.6203,
    "Structure_measure": 0.7753
}

# 定义比较规则：对于 MAE 越小越好，其他指标越大越好
def compare_metrics(metric_name, value1, value2):
    if metric_name == "MAE":
        return "模型 1" if value1 < value2 else "模型 2"
    else:
        return "模型 1" if value1 > value2 else "模型 2"

# 比较两组模型的指标
results = {}
for metric in metrics_model_1.keys():
    results[metric] = compare_metrics(metric, metrics_model_1[metric], metrics_model_2[metric])

# 打印比较结果
print("比较结果：")
for metric, winner in results.items():
    print(f"{metric} 指标更好的模型是: {winner}")

# 统计每个模型在多少指标上获胜
model_1_wins = sum(1 for winner in results.values() if winner == "模型 1")
model_2_wins = sum(1 for winner in results.values() if winner == "模型 2")

print("\n总结：")
print(f"模型 1 在 {model_1_wins} 个指标上表现更好。")
print(f"模型 2 在 {model_2_wins} 个指标上表现更好。")

# 判断哪个模型整体更好
if model_1_wins > model_2_wins:
    print("\n综合来看：模型 1 整体表现更好。")
elif model_1_wins < model_2_wins:
    print("\n综合来看：模型 2 整体表现更好。")
else:
    print("\n综合来看：两个模型表现相当。")
