import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="wide")

# --- 1. Load Trained Model & Data ---
try:
    model = joblib.load("ipl_winner_model.pkl")
    df_matches = pd.read_csv(r"C:\Users\sanke\Downloads\matches.csv")

    # Ensure season is numeric
    df_matches['season'] = pd.to_numeric(df_matches['season'], errors='coerce')
    df_matches.dropna(subset=['season'], inplace=True)
    df_matches['season'] = df_matches['season'].astype(int)

    # Ensure chronological order
    if "date" in df_matches.columns:
        df_matches = df_matches.sort_values(['season', 'date'], ascending=False)
    else:
        df_matches = df_matches.sort_values(['season'], ascending=False)

except FileNotFoundError:
    st.error("Model or data files not found. Please ensure 'ipl_winner_model.pkl' and 'matches.csv' exist.")
    st.stop()

# --- 2. Dropdown Options ---
teams = sorted(df_matches['team1'].unique().tolist())
venues = sorted(df_matches['venue'].unique().tolist())
seasons = sorted(df_matches['season'].unique().tolist(), reverse=True)
toss_decisions = ["bat", "field"]

# --- 3. Streamlit UI ---
st.title("🏏 IPL Match Winner Prediction Dashboard")
st.markdown("Select match details from the sidebar to predict the winner and see team analytics.")

st.sidebar.header("Match Details")
team1 = st.sidebar.selectbox("Select Team 1", options=teams)
team2 = st.sidebar.selectbox("Select Team 2", options=[team for team in teams if team != team1])
venue = st.sidebar.selectbox("Select Venue", options=venues)
toss_winner = st.sidebar.selectbox("Toss Winner", options=[team1, team2])
toss_decision = st.sidebar.selectbox("Toss Decision", options=toss_decisions)
season = st.sidebar.selectbox("Select Season (Year)", options=seasons)

# --- 4. Feature Calculation Functions ---
def calculate_recent_winrate(team, season, df, last_n=5):
    past_matches = df[(df['season'] < season) & ((df['team1'] == team) | (df['team2'] == team))]
    past_matches = past_matches.head(last_n)  # head() since data sorted descending
    if past_matches.empty:
        return 0.5
    wins = (past_matches['winner'] == team).sum()
    return wins / len(past_matches)

def calculate_h2h_wins(team1, team2, df):
    past_matches = df[((df['team1'] == team1) & (df['team2'] == team2)) |
                      ((df['team1'] == team2) & (df['team2'] == team1))]
    team1_wins = (past_matches['winner'] == team1).sum()
    team2_wins = (past_matches['winner'] == team2).sum()
    return team1_wins, team2_wins

def get_venue_performance(team, venue, df):
    team_matches_at_venue = df[((df['team1'] == team) | (df['team2'] == team)) & (df['venue'] == venue)]
    if team_matches_at_venue.empty:
        return 0.5
    wins = (team_matches_at_venue['winner'] == team).sum()
    return wins / len(team_matches_at_venue)

# --- 5. Prediction and Visualization ---
if st.sidebar.button("📊 Predict & Analyze"):
    if team1 == team2:
        st.error("Please select two different teams.")
    else:
        # --- Calculate Features ---
        team1_recent = calculate_recent_winrate(team1, season, df_matches)
        team2_recent = calculate_recent_winrate(team2, season, df_matches)
        team1_h2h_wins, team2_h2h_wins = calculate_h2h_wins(team1, team2, df_matches)
        h2h_diff = team1_h2h_wins - team2_h2h_wins
        team1_venue_perf = get_venue_performance(team1, venue, df_matches)
        team2_venue_perf = get_venue_performance(team2, venue, df_matches)

        # --- Create Input for Model ---
        input_data = pd.DataFrame([{
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'season': season,
            'toss_win_equals_winner': 1,
            'h2h_diff': h2h_diff,
            'team1_recent_winrate': team1_recent,
            'team2_recent_winrate': team2_recent
        }])

        # --- Make Prediction ---
        predicted_winner = model.predict(input_data)[0]

        # ✅ Ensure it's a plain string
        if isinstance(predicted_winner, (np.ndarray, list, tuple, pd.Series)):
            predicted_winner = str(predicted_winner[0])
        else:
            predicted_winner = str(predicted_winner)

        st.write("---")

        # --- Display Results in Columns ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🏆 Prediction")
            st.metric(label="Predicted Winner", value=predicted_winner)
            st.write("---")
            st.subheader("Match Info")
            st.text(f"Season: {season}")
            st.text(f"Venue: {venue}")
            st.text(f"Toss Winner: {toss_winner} ({toss_decision})")

        with col2:
            st.subheader("⚔️ Head-to-Head Record")
            if team1_h2h_wins + team2_h2h_wins == 0:
                st.info("No past matches found between these two teams.")
            else:
                h2h_data = pd.DataFrame({'Team': [team1, team2], 'Wins': [team1_h2h_wins, team2_h2h_wins]})
                fig_h2h = px.pie(h2h_data, names='Team', values='Wins', hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_h2h, use_container_width=True)

        st.write("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📈 Recent Form (Last 5 Matches)")
            form_data = pd.DataFrame({
                'Team': [team1, team2],
                'Win Rate': [team1_recent, team2_recent]
            })
            fig_form = px.bar(form_data, x='Team', y='Win Rate', text_auto='.2f',
                              color='Team', color_discrete_sequence=px.colors.qualitative.Set2)
            fig_form.update_layout(yaxis_title="Win Rate", xaxis_title=None)
            st.plotly_chart(fig_form, use_container_width=True)

        with col4:
            st.subheader(f"🏟️ Performance at {venue}")
            venue_data = pd.DataFrame({
                'Team': [team1, team2],
                'Win Rate': [team1_venue_perf, team2_venue_perf]
            })
            fig_venue = px.bar(venue_data, x='Team', y='Win Rate', text_auto='.2f',
                               color='Team', color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_venue.update_layout(yaxis_title="Win Rate at Venue", xaxis_title=None)
            st.plotly_chart(fig_venue, use_container_width=True)
