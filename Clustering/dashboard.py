# Import necessary libraries
import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit page settings
st.set_page_config(page_title="Apriori Association Rules", layout="wide")
st.title("🛒 Market Basket Analysis with Apriori Algorithm")

# Step 1: Upload the CSV file
uploaded_file = st.file_uploader("📂 Upload 'Groceries_dataset.csv'", type=["csv"])

if uploaded_file:
    # Step 2: Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Step 3: Show a preview of the uploaded data
    st.subheader("📄 Preview of Uploaded Data")
    st.write(df.head())

    # Step 4: Check if required columns are present
    if {'Member_number', 'Date', 'itemDescription'}.issubset(df.columns):
        # Step 5: Group items bought by each customer on a specific date into a single list
        df_grouped = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()

        # Step 6: Convert grouped data into a list of transactions
        transactions = df_grouped['itemDescription'].tolist()

        # Step 7: Apply one-hot encoding to the transaction list
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        st.success("✅ Transactions prepared and encoded.")

        # Step 8: Add sidebar sliders to set parameters
        st.sidebar.header("⚙️ Settings")
        min_support = st.sidebar.slider("Min Support", 0.001, 0.02, 0.002, step=0.001)
        min_confidence = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.1, step=0.05)

        # Step 9: Apply Apriori algorithm to find frequent itemsets
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

        # Step 10: Derive association rules from the frequent itemsets
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if not rules.empty:
            # Step 11: Sort rules by 'lift' and display top 10 rules
            rules_sorted = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']] \
                            .sort_values(by='lift', ascending=False)

            st.subheader("📈 Top 10 Association Rules")
            st.dataframe(rules_sorted.head(10).style.format({
                'support': "{:.3f}",
                'confidence': "{:.2f}",
                'lift': "{:.2f}"
            }))

            # Step 12: Allow user to download all rules as a CSV
            csv = rules_sorted.to_csv(index=False)
            st.download_button("⬇️ Download All Rules as CSV", csv, file_name="association_rules.csv", mime="text/csv")
        else:
            # If no rules found, show warning
            st.warning("⚠️ No association rules found. Try lowering the support or confidence.")
    else:
        # If the dataset is missing required columns, show error
        st.error("❌ Columns missing. Dataset must contain 'Member_number', 'Date', and 'itemDescription'.")
else:
    # Show info message if file not uploaded yet
    st.info("Please upload a CSV file to start.")
