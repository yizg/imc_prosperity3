#!/usr/bin/env python
# coding: utf-8

# ## Expected Profit of Each Container (under different LAMBDA)
# 
# ###  Logit Quantal Response Equilibrium (QRE) — *Game Theory*
# 
# In QRE, instead of always choosing the best option (as in Nash Equilibrium), players respond **probabilistically** to expected payoffs. This accounts for *bounded rationality* — the idea that real-world players aren't perfectly logical.
# 
# The choice probability for container Ci is given by:
# 
# $
# P(i) = \frac{e^{\lambda \cdot U_i}}{\sum_{j} e^{\lambda \cdot U_j}}
# $
# 
# $\lambda$ : Rationality parameter
# 
# Low $ \lambda $: Noisy or random behavior.
# 
# High $ \lambda $: As $\lambda$→∞, players become "perfectly rational", and play approaches a Nash equilibrium. (Wiki)
# 

# In[60]:


import pandas as pd
import numpy as np

BASE_TREASURE = 10000
LAMBDA = 1  # Rationality parameter: higher means more rational players, subject to change manually

containers = pd.DataFrame([
    {"container": "C0", "multiplier": 10, "inhabitants": 1},
    {"container": "C1", "multiplier": 17, "inhabitants": 1},
    {"container": "C2", "multiplier": 20, "inhabitants": 2},
    {"container": "C3", "multiplier": 31, "inhabitants": 2},
    {"container": "C4", "multiplier": 37, "inhabitants": 3},
    {"container": "C5", "multiplier": 50, "inhabitants": 4},
    {"container": "C6", "multiplier": 73, "inhabitants": 4},
    {"container": "C7", "multiplier": 80, "inhabitants": 6},
    {"container": "C8", "multiplier": 89, "inhabitants": 8},
    {"container": "C9", "multiplier": 90, "inhabitants": 10},
])

# utility
containers["utility"] = containers["multiplier"] / containers["inhabitants"]

# quantal response eqbrm(QRE): choice probability (softmax with lambda)
def quantal_response(utility, lam):
    scaled_util = lam * utility
    exp_util = np.exp(scaled_util - np.max(scaled_util))  # numerical stability
    return exp_util / exp_util.sum()

containers["popularity_prob"] = quantal_response(containers["utility"], LAMBDA)

containers["popularity_factor"] = containers["popularity_prob"] * 100

# expected profit
containers["expected_profit"] = (BASE_TREASURE * containers["multiplier"]) / (
    containers["inhabitants"] + containers["popularity_factor"]
)

containers["rank"] = containers["expected_profit"].rank(ascending=False)
containers_sorted = containers.sort_values(by="expected_profit", ascending=False)
containers_sorted[["container", "multiplier", "inhabitants", "utility", "popularity_factor", "expected_profit", "rank"]]


# ## Top Containers with different lambda

# In[50]:


import matplotlib.pyplot as plt

lambda_values = np.linspace(0.01, 1, 80) #(start value, end value, number of points), subject to change manually
top3_tracking = {container_id: [] for container_id in containers["container"]}

# visualise the top picks and LAMBDA
records = []

# each lambda, compute and store top containers (with rank)
for lam in lambda_values:
    utility = containers["multiplier"] / containers["inhabitants"]
    popularity_prob = quantal_response(utility, lam)
    popularity_factor = popularity_prob * 100
    expected_profit = (BASE_TREASURE * containers["multiplier"]) / (
        containers["inhabitants"] + popularity_factor
    )
    sorted_indices = expected_profit.sort_values(ascending=False).index[:7]
    for rank, idx in enumerate(sorted_indices, start=1):
        records.append({
            "lambda": lam,
            "rank": rank,
            "container_id": containers.loc[idx, "container"]
        })


top_containers_by_lambda = pd.DataFrame.from_records(records)

# Plot
import seaborn as sns
plt.figure(figsize=(20, 8))
sns.scatterplot(
    data=top_containers_by_lambda,
    x="lambda",
    y="rank",
    hue="container_id",
    palette="tab10",
    s=100
)
plt.gca().invert_yaxis()
plt.xlabel("Lambda (Rationality Level)")
plt.ylabel("Rank (1=Best)")
plt.title("Top 7 Containers Across Different Rationality Levels")
plt.legend(title="Container ID", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Profit (with corrisponding container)  vs Lambda

# In[64]:


from itertools import combinations

BASE_TREASURE = 10000
CONTAINER_COST = 50000
LAMBDA_VALUES = np.linspace(0.001, 0.1, 40)  #(start value, end value, number of points), subject to change manually


results = []

# Iterate over different lambda values
for lam in LAMBDA_VALUES:
    containers["utility"] = containers["multiplier"] / containers["inhabitants"]
    containers["popularity_prob"] = quantal_response(containers["utility"], lam)
    containers["popularity_factor"] = containers["popularity_prob"] * 100
    
    # expected profit for each container
    containers["expected_profit"] = (BASE_TREASURE * containers["multiplier"]) / (
        containers["inhabitants"] + containers["popularity_factor"]
    )
    
    # 2-container combinations
    best_total_profit = -np.inf
    best_combo = ()

    for combo in combinations(containers.index, 2):
        profit1 = containers.loc[combo[0], "expected_profit"]
        profit2 = containers.loc[combo[1], "expected_profit"] - CONTAINER_COST
        total = profit1 + profit2

        if total > best_total_profit:
            best_total_profit = total
            best_combo = (containers.loc[combo[0], "container"], containers.loc[combo[1], "container"])

    results.append({
        "lambda": lam,
        "total_profit": best_total_profit,
        "container_1": best_combo[0],
        "container_2": best_combo[1]
    })


results_df = pd.DataFrame(results)


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results_df["lambda"], results_df["total_profit"], marker='o')
ax.set_xlabel("Lambda (λ)")
ax.set_ylabel("Estimated Total Profit")
ax.set_title("Estimated Total Profit vs Lambda (with 50k cost for second container)")
ax.grid(True)

# Annotate best choices
for i in range(0, len(results_df), 5):
    label = f"{results_df['container_1'][i]} + {results_df['container_2'][i]}"
    ax.annotate(label,
                (results_df["lambda"][i], results_df["total_profit"][i]),
                textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

plt.tight_layout()
plt.show()


# **If $\lambda$ tends to 0, ie, probility of choose every container is the same. The best option is C6:multiper 73, and C7:multiper 80, expected profit around 52,140**

# ![Screenshot%202025-04-11%20at%206.20.43%20pm.png](attachment:Screenshot%202025-04-11%20at%206.20.43%20pm.png)

# ![Screenshot%202025-04-11%20at%206.20.35%20pm.png](attachment:Screenshot%202025-04-11%20at%206.20.35%20pm.png)

# In[ ]:




