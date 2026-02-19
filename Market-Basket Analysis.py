import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.markdown(
    "<h1 style='text-align:center;color:#FF5722;'>🛒 Market Basket Analysis (Apriori)</h1>",
    unsafe_allow_html=True
)

# ----------------------------------------
# 1. Sample Transactions
# ----------------------------------------
transactions = [
    ["milk", "bread", "eggs"],
    ["bread", "butter"],
    ["milk", "bread"],
    ["butter", "eggs"],
    ["milk", "eggs"],
]

st.subheader("📦 Sample Transactions")
st.write(transactions)

# ----------------------------------------
# 2. User Controls
# ----------------------------------------
min_support = st.slider("Minimum Support", 0.1, 1.0, 0.4)
min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)

# ----------------------------------------
# 3. Item Frequency Plot
# ----------------------------------------
st.subheader("📊 Item Frequency")

flat_items = [item for transaction in transactions for item in transaction]
item_counts = Counter(flat_items)

fig1, ax1 = plt.subplots()
ax1.bar(item_counts.keys(), item_counts.values())
ax1.set_xlabel("Items")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

# ----------------------------------------
# 4. Encode Transactions
# ----------------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# ----------------------------------------
# 5. Apply Apriori
# ----------------------------------------
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
frequent_itemsets['support'] = frequent_itemsets['support'].round(2)

st.subheader("📊 Frequent Itemsets")
st.dataframe(frequent_itemsets)

# ----------------------------------------
# 6. Association Rules
# ----------------------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

if not rules.empty:
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules[['support', 'confidence', 'lift']] = rules[['support', 'confidence', 'lift']].round(2)

    st.subheader("🔗 Association Rules")
    st.dataframe(rules)

    # ----------------------------------------
    # 7. Lift vs Confidence Plot
    # ----------------------------------------
    st.subheader("📈 Lift vs Confidence")

    fig2, ax2 = plt.subplots()
    ax2.scatter(rules['confidence'], rules['lift'])
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Lift")
    st.pyplot(fig2)

    # ----------------------------------------
    # 8. Network Graph
    # ----------------------------------------
    st.subheader("🌐 Association Network Graph")

    G = nx.DiGraph()

    for _, row in rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'])

    fig3, ax3 = plt.subplots()
    pos = nx.spring_layout(G, k=0.6)
    nx.draw(G, pos, with_labels=True,
            node_color='lightblue',
            node_size=2000,
            font_size=10,
            arrows=True,
            ax=ax3)

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels={k: f"{v:.2f}" for k, v in labels.items()},
                                 ax=ax3)

    st.pyplot(fig3)

else:
    st.warning("No association rules found for selected support/confidence values.")
