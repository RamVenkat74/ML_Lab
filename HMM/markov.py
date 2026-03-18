import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

st.title("Weather Prediction using Markov Model")

st.header("1. Dataset Preview")
# Note: Ensure the file path is correct relative to where you run the script
df = pd.read_csv("weatherHistory.csv")
st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

st.header("2. Preparing Markov States")
# Extract top 6 most frequent weather conditions to keep the model simple
weather_seq = df["Summary"].astype(str)
top_states = weather_seq.value_counts().head(6).index
weather_seq = weather_seq[weather_seq.isin(top_states)]
states = weather_seq.unique()

# Create mappings between state names and matrix indices
state_to_idx = {s: i for i, s in enumerate(states)}
idx_to_state = {i: s for s, i in state_to_idx.items()}
sequence = weather_seq.map(state_to_idx).values

st.write("States Used:", list(states))

st.header("3. Training Markov Model")
n_states = len(states)
transition_matrix = np.zeros((n_states, n_states))

# Count transitions from state i to state j
for i in range(len(sequence) - 1):
    transition_matrix[sequence[i], sequence[i+1]] += 1

# Normalize rows to convert counts into probabilities
# (Adding a small epsilon or handling division by zero is recommended for production)
transition_matrix = transition_matrix / \
    transition_matrix.sum(axis=1, keepdims=True)

st.write("Transition Matrix")
st.dataframe(pd.DataFrame(transition_matrix, index=states, columns=states))

st.header("4. State Transition Visualization")
G = nx.DiGraph()

# Build the graph, only adding edges for probabilities > 5% to reduce clutter
for i in range(n_states):
    for j in range(n_states):
        prob = round(transition_matrix[i][j], 2)
        if prob > 0.05:
            G.add_edge(idx_to_state[i], idx_to_state[j], weight=prob)

pos = nx.circular_layout(G)
edge_labels = nx.get_edge_attributes(G, 'weight')

fig, ax = plt.subplots(figsize=(8, 6))
nx.draw(G, pos, ax=ax, with_labels=True,
        node_size=2500, node_color="lightblue")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
st.pyplot(fig)

st.header("5. Predict Next Weather")
current_state = st.selectbox("Select Current Weather Condition", states)

if st.button("Predict Next State"):
    idx = state_to_idx[current_state]
    probs = transition_matrix[idx]
    # Make a weighted random choice based on the transition probabilities
    next_state_idx = np.random.choice(len(probs), p=probs)
    predicted = idx_to_state[next_state_idx]
    st.success(f"Predicted Next Weather: {predicted}")

st.header("6. Model Evaluation")
correct = 0
total = 0

# Evaluate by predicting the most likely next state (argmax)
for i in range(len(sequence) - 1):
    current = sequence[i]
    predicted = np.argmax(transition_matrix[current])
    actual = sequence[i+1]

    if predicted == actual:
        correct += 1
    total += 1

accuracy = correct / total
st.write("Prediction Accuracy:", round(accuracy * 100, 2), "%")
