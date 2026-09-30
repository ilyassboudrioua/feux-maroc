import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Calendrier des feux - Maroc", layout="wide")
st.title("Calendrier des feux au Maroc (2001-2024)")
st.caption("Données : NASA FIRMS (MODIS et VIIRS S-NPP), harmonisées en équivalent VIIRS.")

NOMS_MOIS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
             "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

# Zones d'intérêt (rectangles approximatifs : lat_min, lat_max, lon_min, lon_max)
ZONES = {
    "Tout le Maroc": None,
    "Nord et Rif": (34.3, 36.0, -6.5, -2.0),
    "Moyen Atlas": (32.5, 34.0, -6.0, -3.5),
    "Tadla - Beni Mellal": (32.0, 32.8, -7.2, -6.0),
    "Souss-Massa": (29.0, 31.0, -10.0, -7.0),
}


@st.cache_data
def charger():
    feux = pd.read_csv("data/feux_maroc_propres.csv", parse_dates=["date"])
    facteurs = pd.read_csv("data/facteurs_mois.csv", index_col=0).iloc[:, 0]
    return feux, facteurs


def filtrer_zone(feux, zone):
    box = ZONES[zone]
    if box is None:
        return feux
    lat_min, lat_max, lon_min, lon_max = box
    return feux[feux["latitude"].between(lat_min, lat_max) &
                feux["longitude"].between(lon_min, lon_max)]


def harmoniser(feux_zone, facteurs):
    grille = pd.MultiIndex.from_product([range(2001, 2025), range(1, 13)], names=["annee", "mois"])
    m = (feux_zone.groupby([feux_zone["date"].dt.year.rename("annee"),
                            feux_zone["date"].dt.month.rename("mois"), "capteur"])
         .size().unstack(fill_value=0)
         .reindex(grille, fill_value=0))
    for col in ["MODIS", "VIIRS"]:
        if col not in m:
            m[col] = 0
    m = m.reset_index()
    m["MODIS_corrige"] = m["MODIS"] * m["mois"].map(facteurs)
    m["harmonise"] = np.where(m["annee"] >= 2012, m["VIIRS"], m["MODIS_corrige"])
    return m


feux, facteurs = charger()

# Filtres
st.sidebar.header("Filtres")
zone = st.sidebar.selectbox("Zone d'intérêt", list(ZONES.keys()))
annees = st.sidebar.slider("Période", 2001, 2024, (2003, 2024))

feux_zone = filtrer_zone(feux, zone)
serie = harmoniser(feux_zone, facteurs)
serie_f = serie[serie["annee"].between(*annees)]

if annees[0] < 2012:
    st.info("Avant 2012, les valeurs sont estimées à partir de MODIS (VIIRS n'existait pas). "
            "2001 et 2002 sont incomplètes : un seul satellite MODIS était en service.")

# Chiffres clés
par_an = serie_f.groupby("annee")["harmonise"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("Feux (équivalent VIIRS)", f"{par_an.sum():,.0f}".replace(",", " "))
c2.metric("Année record", int(par_an.idxmax()) if par_an.sum() > 0 else "-")
mois_pic = serie_f.groupby("mois")["harmonise"].sum().idxmax()
c3.metric("Mois le plus actif", NOMS_MOIS[mois_pic - 1])

# Évolution annuelle
st.subheader(f"Évolution annuelle - {zone}")
df_an = par_an.reset_index()
df_an["source"] = np.where(df_an["annee"] >= 2012, "Mesuré (VIIRS)", "Estimé (MODIS corrigé)")
fig = px.line(df_an, x="annee", y="harmonise", markers=True, color="source",
              labels={"annee": "Année", "harmonise": "Feux par an", "source": "Source"})
if annees[0] <= 2012 <= annees[1]:
    fig.add_vline(x=2012, line_dash="dot", annotation_text="Arrivée de VIIRS")
st.plotly_chart(fig, use_container_width=True)

# Calendrier
st.subheader("Calendrier des feux")
cal = serie_f.pivot_table(index="annee", columns="mois", values="harmonise", aggfunc="sum")
fig_cal = px.imshow(cal, x=NOMS_MOIS, color_continuous_scale="YlOrRd", aspect="auto",
                    labels={"x": "Mois", "y": "Année", "color": "Feux"})
st.plotly_chart(fig_cal, use_container_width=True)

# Anomalies
st.subheader("Mois anormaux (score z)")
base = serie_f[serie_f["annee"] >= 2003].copy()
stats = base.groupby("mois")["harmonise"].agg(["mean", "std"])
base = base.join(stats, on="mois")
base["z"] = ((base["harmonise"] - base["mean"]) / base["std"]).fillna(0)
carte_z = base.pivot(index="annee", columns="mois", values="z")
fig_z = px.imshow(carte_z, x=NOMS_MOIS, color_continuous_scale="RdBu_r",
                  zmin=-3, zmax=3, aspect="auto",
                  labels={"x": "Mois", "y": "Année", "color": "Score z"})
st.plotly_chart(fig_z, use_container_width=True)

top = base.sort_values("z", ascending=False).head(5)
top["mois"] = top["mois"].map(lambda i: NOMS_MOIS[i - 1])
st.write("Les 5 mois les plus anormaux :")
st.dataframe(top[["annee", "mois", "harmonise", "z"]].round(1), hide_index=True)

# Carte
st.subheader("Carte des détections")
annee_carte = st.selectbox("Année à afficher sur la carte", list(range(2024, 2000, -1)))
st.map(feux_zone[feux_zone["date"].dt.year == annee_carte],
       latitude="latitude", longitude="longitude", size=300)