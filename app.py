import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore Rental Fairness Analyzer",
    page_icon="🏠",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/bangalore_rent_scored.csv')
    ranking = pd.read_csv('data/locality_ranking.csv')
    return df, ranking

df, ranking = load_data()

# ── Retrain model ─────────────────────────
@st.cache_resource
def train_model(df):
    df_model = df.copy()
    bhk_map = {'1BHK': 1, '2BHK': 2, '3BHK': 3}
    df_model['BHK'] = df_model['HouseType'].map(bhk_map)
    le = LabelEncoder()
    df_model['Locality_encoded'] = le.fit_transform(df_model['Locality'])
    X = df_model[['BHK', 'Locality_encoded']]
    y = df_model['AvgRent']
    model = LinearRegression()
    model.fit(X, y)
    return model, le

model, le = train_model(df)

# ── Fairness score function ───────────────────────────────────
def calculate_fairness_score(actual, predicted, zscore):
    pct_diff = (actual - predicted) / predicted * 100
    base_score = 100 - max(0, pct_diff)
    zscore_penalty = max(0, zscore * 5)
    return round(max(0, min(100, base_score - zscore_penalty)), 1)

def score_label(score):
    if score >= 80:   return '✅ Fair Price'
    elif score >= 60: return '⚠️ Slightly High'
    elif score >= 40: return '🔶 Overpriced'
    else:             return '🚨 Significantly Overpriced'

def score_color(score):
    if score >= 80:   return 'green'
    elif score >= 60: return 'orange'
    elif score >= 40: return 'orange'
    else:             return 'red'

# ── Coordinates dictionary ────────────────────────────────────
locality_coords = {
    'HSR Layout': (12.9081, 77.6476),
    'BTM Layout': (12.9166, 77.6101),
    'Koramangala': (12.9352, 77.6245),
    'JP Nagar Phase 7': (12.8925, 77.5831),
    'Whitefield': (12.9698, 77.7499),
    'Electronic City Phase 1': (12.8399, 77.6770),
    'Electronics City': (12.8458, 77.6692),
    'Indiranagar': (12.9784, 77.6408),
    'Marathahalli': (12.9591, 77.6974),
    'Bellandur': (12.9257, 77.6761),
    'Bannerughatta': (12.8635, 77.5970),
    'Banaswadi': (13.0104, 77.6553),
    'Basaveswarnagar': (12.9810, 77.5345),
    'CV Raman Nagar': (12.9850, 77.6600),
    'Carmelaram': (12.8918, 77.7218),
    'Cooke Town': (13.0003, 77.6122),
    'Doddakammanahalli': (12.8812, 77.6401),
    'Amruthahalli': (13.0480, 77.5900),
    'Akshayanagar': (12.8629, 77.6211),
    'Frazer Town': (12.9888, 77.6200),
    'Domlur': (12.9607, 77.6387),
    'Richmond Town': (12.9636, 77.5985),
    'Shanti Nagar': (12.9565, 77.5953),
    'Rajajinagar': (12.9902, 77.5524),
    'Hebbal': (13.0358, 77.5970),
    'Yeshwanthpur': (13.0234, 77.5456),
    'Malleshwaram': (13.0034, 77.5678),
    'Jayanagar': (12.9253, 77.5832),
    'Banashankari': (12.9253, 77.5634),
    'JP Nagar': (12.9078, 77.5833),
    'Kanakapura Road': (12.9012, 77.5456),
    'Electronic City Phase 2': (12.8312, 77.6756),
    'Bommasandra': (12.8234, 77.6934),
    'Sarjapur': (12.8596, 77.7848),
    'Varthur': (12.9376, 77.7479),
    'Marathahalli': (12.9591, 77.6974),
    'KR Puram': (13.0034, 77.6934),
    'Hennur': (13.0345, 77.6234),
    'Nagawara': (13.0456, 77.6234),
    'Hoodi': (12.9823, 77.7123),
    'Yelahanka': (13.1004, 77.5963),
    'Devanahalli': (13.2462, 77.7131),
    'Hoskote': (13.0709, 77.7986),
    'Hulimavu': (12.8723, 77.6178),
    'Bommanahalli': (12.8996, 77.6394),
    'Begur': (12.8619, 77.6273),
    'Gottigere': (12.8466, 77.6012),
    'Chandapura': (12.8036, 77.6841),
}

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🏠 Bangalore Rental Fairness Analyzer")
st.markdown("*Am I being overcharged? Find out instantly.*")
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — FAIRNESS CHECKER
# ══════════════════════════════════════════════════════════════
st.header("🔍 Check Any Listing")

col1, col2, col3 = st.columns(3)

with col1:
    localities = sorted(df['Locality'].unique().tolist())
    selected_locality = st.selectbox("Select Locality", localities)

with col2:
    selected_bhk = st.selectbox("BHK Type", ['1BHK', '2BHK', '3BHK'])

with col3:
    actual_rent = st.number_input(
        "Your Quoted Rent (₹)",
        min_value=1000,
        max_value=200000,
        value=15000,
        step=500
    )

if st.button("🔎 Analyse This Listing", use_container_width=True):
    bhk_map = {'1BHK': 1, '2BHK': 2, '3BHK': 3}
    bhk_num = bhk_map[selected_bhk]

    if selected_locality in le.classes_:
        loc_encoded = le.transform([selected_locality])[0]
        predicted_rent = model.predict([[bhk_num, loc_encoded]])[0]
    else:
        predicted_rent = df[df['HouseType'] == selected_bhk]['AvgRent'].mean()

    loc_data = df[df['Locality'] == selected_locality]['AvgRent']
    loc_mean = loc_data.mean()
    loc_std = loc_data.std() if loc_data.std() > 0 else 1
    zscore = (actual_rent - loc_mean) / loc_std

    score = calculate_fairness_score(actual_rent, predicted_rent, zscore)
    verdict = score_label(score)
    color = score_color(score)

    st.divider()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Your Rent", f"₹{actual_rent:,}")
    r2.metric("Fair Rent Estimate", f"₹{predicted_rent:,.0f}",
              delta=f"₹{actual_rent - predicted_rent:,.0f}")
    r3.metric("Fairness Score", f"{score}/100")
    r4.metric("Verdict", verdict)

    st.markdown(f"**Fairness Score: {score}/100**")
    st.progress(int(score) / 100)

    diff = actual_rent - predicted_rent
    if diff > 0:
        st.warning(f"This listing is **₹{diff:,.0f} above** the predicted fair rent for "
                   f"a {selected_bhk} in {selected_locality}.")
    else:
        st.success(f"Great deal! This listing is **₹{abs(diff):,.0f} below** the predicted "
                   f"fair rent. You're saving money. 💚")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — LOCALITY COMPARISON
# ══════════════════════════════════════════════════════════════
st.header("📊 Locality Comparison")

compare_localities = st.multiselect(
    "Select localities to compare",
    options=sorted(df['Locality'].unique().tolist()),
    default=['HSR Layout', 'BTM Layout', 'Koramangala', 'JP Nagar Phase 7', 'Electronics City']
)

if compare_localities:
    compare_df = ranking[ranking['Locality'].isin(compare_localities)][
        ['Locality', 'avg_rent', 'avg_fairness', 'pct_overpriced', 'rank']
    ].sort_values('avg_fairness', ascending=False)

    compare_df.columns = ['Locality', 'Avg Rent (₹)', 'Fairness Score', '% Overpriced Listings', 'City Rank']
    compare_df['Avg Rent (₹)'] = compare_df['Avg Rent (₹)'].apply(lambda x: f'₹{x:,.0f}')
    compare_df['Fairness Score'] = compare_df['Fairness Score'].apply(lambda x: f'{x:.1f}/100')

    st.dataframe(compare_df, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — INTERACTIVE MAP
# ══════════════════════════════════════════════════════════════
st.header("🗺️ Bangalore Rental Fairness Map")
st.caption("Green = Fair Price · Orange = Slightly High · Red = Overpriced · Click any dot for details")

def get_color(score):
    if score >= 80:   return '#4CAF50'
    elif score >= 60: return '#FFC107'
    elif score >= 40: return '#FF9800'
    else:             return '#F44336'

m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles='CartoDB positron')

for _, row in ranking.iterrows():
    coords = locality_coords.get(row['Locality'])
    if not coords:
        continue
    folium.CircleMarker(
        location=coords,
        radius=10,
        color=get_color(row['avg_fairness']),
        fill=True,
        fill_color=get_color(row['avg_fairness']),
        fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>{row['Locality']}</b><br>"
            f"Avg Rent: ₹{row['avg_rent']:,.0f}<br>"
            f"Score: {row['avg_fairness']:.1f}/100<br>"
            f"Rank: #{int(row['rank'])}",
            max_width=200
        ),
        tooltip=f"{row['Locality']} — {row['avg_fairness']:.0f}/100"
    ).add_to(m)

st_folium(m, width=None, height=500)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — TOP & BOTTOM LOCALITIES
# ══════════════════════════════════════════════════════════════
st.header("🏆 City-wide Rankings")

tab1, tab2 = st.tabs(["💚 Best Value Areas", "🚨 Most Overpriced Areas"])

with tab1:
    top = ranking.head(15)[['rank', 'Locality', 'avg_rent', 'avg_fairness']].copy()
    top.columns = ['Rank', 'Locality', 'Avg Rent (₹)', 'Fairness Score']
    top['Avg Rent (₹)'] = top['Avg Rent (₹)'].apply(lambda x: f'₹{x:,.0f}')
    top['Fairness Score'] = top['Fairness Score'].apply(lambda x: f'{x:.1f}/100')
    st.dataframe(top, use_container_width=True, hide_index=True)

with tab2:
    bottom = ranking.tail(15).sort_values('avg_fairness')[
        ['rank', 'Locality', 'avg_rent', 'avg_fairness']
    ].copy()
    bottom.columns = ['Rank', 'Locality', 'Avg Rent (₹)', 'Fairness Score']
    bottom['Avg Rent (₹)'] = bottom['Avg Rent (₹)'].apply(lambda x: f'₹{x:,.0f}')
    bottom['Fairness Score'] = bottom['Fairness Score'].apply(lambda x: f'{x:.1f}/100')
    st.dataframe(bottom, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Built by Yayavari R · Bangalore Rental Fairness Analyzer · "
           "Data: Kaggle · Model: Linear Regression + Z-score Anomaly Detection")