from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COLORS = {"Developmentally apparent":"#4f8f88","Complex multi-condition":"#6f77a8","Less-obvious overlapping":"#e48763"}
ORDER = list(COLORS)
PROFILE_DESCRIPTIONS = {
    "Developmentally apparent": "More reported speech and developmental differences",
    "Complex multi-condition": "Multiple reported developmental and behavioral conditions",
    "Less-obvious overlapping": "Fewer developmental delays; more ADHD and anxiety",
}

st.set_page_config(page_title="When Autism Isn't Obvious", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1120px;padding-top:2rem}.hero{background:linear-gradient(120deg,#eef8f6,#fff4ed);border-radius:24px;padding:34px;margin-bottom:24px}.eyebrow{color:#247a73;font-weight:750;letter-spacing:.08em;text-transform:uppercase;font-size:.78rem}.hero h1{color:#17324d;font-size:2.55rem;line-height:1.08;margin:.35rem 0 .8rem}.hero p{font-size:1.16rem;color:#42566b;max-width:850px}.profile{background:white;border:1px solid #e4eaed;border-radius:18px;padding:20px;min-height:290px;box-shadow:0 3px 12px rgba(35,55,70,.05)}.profile h3{font-size:1.17rem;margin:0 0 9px;color:#17324d}.profile-subtitle{color:#617387;font-size:.86rem;line-height:1.3;min-height:45px;margin-bottom:10px}.age{font-size:2.25rem;font-weight:800;color:#17324d}.caption{color:#617387;font-size:.88rem}.callout{border-left:5px solid #e48763;background:#fff6f1;padding:19px 23px;border-radius:9px;margin:20px 0;font-size:1.05rem}.icon-grid{display:grid;grid-template-columns:repeat(10,1fr);gap:7px;max-width:530px;margin:18px auto}.child{height:29px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:15px}.later{background:#e48763;color:white}.earlier{background:#dce9e7;color:#617876}.note{color:#65778a;font-size:.85rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_results():
    return (pd.read_csv(RESULTS/"cluster_profile_summary.csv"), pd.read_csv(RESULTS/"cluster_feature_summary.csv"), pd.read_csv(RESULTS/"cluster_year_summary.csv",dtype={"survey_year":str}))

def icon_grid(rate):
    n=round(rate); cells=[f'<div class="child {"later" if i<n else "earlier"}">●</div>' for i in range(100)]
    return '<div class="icon-grid">'+''.join(cells)+'</div>'

profiles,features,years=load_results()
profiles["profile"]=pd.Categorical(profiles.profile,ORDER,ordered=True); profiles=profiles.sort_values("profile")
st.markdown("""<div class="hero"><div class="eyebrow">Explainable machine learning · 2022–2024 NSCH</div><h1>When autism isn't obvious, answers may come later.</h1><p>Machine learning identified three recurring reported profiles without seeing diagnosis age. Only afterward did we compare when children in each profile first received an autism diagnosis.</p></div>""",unsafe_allow_html=True)

for column,name in zip(st.columns(3),ORDER):
    row=profiles[profiles.profile==name].iloc[0]
    with column:
        st.markdown(f"""<div class="profile" style="border-top:6px solid {COLORS[name]}"><h3>{name}</h3><div class="profile-subtitle">{PROFILE_DESCRIPTIONS[name]}</div><div class="age">Age {row.weighted_median_diagnosis_age:.0f}</div><div class="caption">weighted median first reported diagnosis age</div><br><b>{row.weighted_later_pct:.1f}%</b> diagnosed after age 4<br><span class="caption">{row.sample_n:,.0f} survey records</span></div>""",unsafe_allow_html=True)

st.caption("Profile names summarize caregiver-reported patterns and are descriptive, not recognized clinical autism subtypes.")

st.markdown("""<div class="callout"><b>The discovery:</b> The less-obvious overlapping profile had a weighted median diagnosis age of 7—four years later than the developmentally apparent profile.</div>""",unsafe_allow_html=True)
st.subheader("What characterizes each reported profile?")
chosen=["ADHD","Anxiety","Speech disorder","Developmental delay","Learning disability","Behavior problems"]
pivot=features[features.feature.isin(chosen)].pivot(index="feature",columns="profile",values="weighted_prevalence_pct")[ORDER]
st.dataframe(pivot.style.format("{:.0f}%").background_gradient(cmap="Blues",axis=None,vmin=0,vmax=100),width="stretch")
st.caption("The profiles are exploratory combinations of caregiver-reported characteristics. They are not clinical autism subtypes.")

st.divider(); st.subheader("Experience the difference")
focus=st.radio("Choose a profile",ORDER,horizontal=True,index=2); row=profiles[profiles.profile==focus].iloc[0]
st.markdown(icon_grid(row.weighted_later_pct),unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center'><b>{row.weighted_later_pct:.1f}%</b> of the weighted population represented by the <b>{focus.lower()}</b> profile received a first reported diagnosis after age 4.</p>",unsafe_allow_html=True)

st.divider(); st.subheader("Did the pattern persist across years?")
yp=years.pivot(index="survey_year",columns="profile",values="weighted_later_pct")[ORDER]
st.line_chart(yp,color=[COLORS[p] for p in ORDER]); st.dataframe(yp.style.format("{:.1f}%"),width="stretch")
st.markdown("<div class='callout'><b>Yes.</b> The less-obvious overlapping profile had the highest later-diagnosis rate in 2022, 2023 and 2024.</div>",unsafe_allow_html=True)

st.divider()
st.caption("Exploratory associations from caregiver-reported survey data; these profiles are not clinical subtypes and do not explain why an individual child was diagnosed later.")
with st.expander("Methods and limitations"):
    st.markdown("""- **Cohort:** 4,759 children aged 5–17 with caregiver-reported autism.

- **Clustering method:** K-modes grouped children with similar categorical patterns, such as whether ADHD, anxiety, speech disorder or developmental delay was reported.

- **Why diagnosis age was excluded:** Diagnosis age was hidden from the algorithm so that any age differences discovered afterward were not built into the groups.

- **Limitation:** Characteristics were reported at survey and were not necessarily present or recognized before diagnosis.""")
st.markdown("<p class='note'>Source: 2022–2024 National Survey of Children's Health, U.S. Census Bureau / HRSA Maternal and Child Health Bureau.</p>",unsafe_allow_html=True)
