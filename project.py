import streamlit as st
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bson import ObjectId
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode


client = MongoClient("mongodb://localhost:27017/")
db = client["agriculture_climate_db"]


db_collections = {
    "Countries and Regions": db.countries_and_regions,
    "Climate": db.climate,
    "Crops": db.crops,
    "Productivity and Economic Impact": db.productivity_and_economic_impact
}

st.set_page_config(page_title="Agriculture DB Admin", layout="wide")


st.sidebar.title("🌾 Climate Change Impact on Agriculture")

st.sidebar.markdown("---")
st.sidebar.markdown("**Prepared by:**")
st.sidebar.markdown("""
- David Maged
- Hatem AbdelMohsen  
- Kerolos Emad  
- Kerolos Nabil  
- Mahmoud Ali  
- Tadros Gamal
""")


menu = st.sidebar.selectbox("Select Action", ["Introduction","Parsing","Insert","Update", "Select","Search","Delete","Indexing","Aggregation","MapReduce","Predict"])

def get_country_region_options():
    data = db.countries_and_regions.find()
    return [f"{doc['country']} - {doc['region']}" for doc in data]

def get_country_region_id(country_region_str):
    country, region = country_region_str.split(" - ")
    result = db.countries_and_regions.find_one({"country": country, "region": region})
    return result["_id"] if result else None

def get_year_options():
    data = db.climate.find()
    return sorted(set(doc['year'] for doc in data), reverse=True)


def get_crop_type_options():
    data = db.crops.find()
    return sorted(set(doc['crop_type'] for doc in data))


def get_productivity_by_year():
    data = db.productivity_and_economic_impact.find()
    return sorted(set(doc['year'] for doc in data), reverse=True)


if menu == "Introduction":
    st.subheader("🌍 Climate & Agriculture Data Explorer")

    st.image("agriculturejpg.jpg", use_container_width=True)

    st.markdown("""
    Welcome to the **Climate and Agriculture Data Explorer**!

    This interactive dashboard is designed to study the **impact of climate on crop productivity around the world**, 
    helping us understand how climate variations affect agricultural outputs and economic performance.

    The system allows users to:
    - **Explore relationships** between climate conditions, crop yields, and economic indicators across countries and regions.
    - **Add, update, or delete data** to keep the system current and accurate.
    - **Analyze statistics** and visualize patterns for informed agricultural planning and policy-making.

    Whether you're a researcher, policymaker, or simply curious, this platform provides valuable insights into the dynamic world of agriculture and climate.
    """)



elif menu == "Parsing":
    st.subheader("📦 Database Tables & Relationships")
    st.markdown("### 🗂️ Database Structure")
    st.markdown("""
    **1. countries_and_regions**
    - `country` (str)
    - `region` (str)

    **2. climate**
    - `year` (int)
    - `average_temperature_c` (float)
    - `total_precipitation_mm` (float)
    - `co2_emissions_mt` (float)
    - `extreme_weather_events` (str)
    - `country_region_id` (ObjectId → countries_and_regions)

    **3. crops**
    - `crop_type` (str)
    - `irrigation_access_percent` (float)
    - `pesticide_use_kg_per_ha` (float)
    - `fertilizer_use_kg_per_ha` (float)
    - `soil_health_index` (float)
    - `adaptation_strategies` (str)
    - `country_region_id` (ObjectId → countries_and_regions)
    - `climate_id` (ObjectId → climate)

    **4. productivity_and_economic_impact**
    - `crop_yield_mt_per_ha` (float)
    - `economic_impact_million_usd` (float)
    - `country_region_id` (ObjectId → countries_and_regions)
    - `climate_id` (ObjectId → climate)
    - `crop_id` (ObjectId → crops)
    """)

    if st.button("🚀 Create Tables and Relationships"):
        df = pd.read_csv('climate_change_impact_on_agriculture_2024.csv')
        client = MongoClient('mongodb://localhost:27017/')
        db = client['agriculture_climate_db']

        countries_collection = db['countries_and_regions']
        climate_collection = db['climate']
        crops_collection = db['crops']
        productivity_collection = db['productivity_and_economic_impact']

        country_region_cache = {}

        for index, row in df.iterrows():
            country = row['Country']
            region = row['Region']
            country_region_key = (country, region)

            if country_region_key in country_region_cache:
                country_region_id = country_region_cache[country_region_key]
            else:
                country_region_doc = {
                    'country': country,
                    'region': region
                }
                inserted = countries_collection.insert_one(country_region_doc)
                country_region_id = inserted.inserted_id
                country_region_cache[country_region_key] = country_region_id

            # climate
            climate_doc = {
                'year': int(row['Year']),
                'average_temperature_c': row['Average_Temperature_C'],
                'total_precipitation_mm': row['Total_Precipitation_mm'],
                'co2_emissions_mt': row['CO2_Emissions_MT'],
                'extreme_weather_events': row['Extreme_Weather_Events'],
                'country_region_id': country_region_id
            }
            climate_id = climate_collection.insert_one(climate_doc).inserted_id

            # crop
            crop_doc = {
                'crop_type': row['Crop_Type'],
                'irrigation_access_percent': row['Irrigation_Access_%'],
                'pesticide_use_kg_per_ha': row['Pesticide_Use_KG_per_HA'],
                'fertilizer_use_kg_per_ha': row['Fertilizer_Use_KG_per_HA'],
                'soil_health_index': row['Soil_Health_Index'],
                'adaptation_strategies': row['Adaptation_Strategies'],
                'country_region_id': country_region_id,
                'climate_id': climate_id
            }
            crop_id = crops_collection.insert_one(crop_doc).inserted_id

            # productivity
            productivity_doc = {
                'crop_yield_mt_per_ha': row['Crop_Yield_MT_per_HA'],
                'economic_impact_million_usd': row['Economic_Impact_Million_USD'],
                'country_region_id': country_region_id,
                'climate_id': climate_id,
                'crop_id': crop_id
            }
            productivity_collection.insert_one(productivity_doc)

        st.success("✅ Data inserted into MongoDB and relationships established!")


# ------------------ INSERT -------------------
if menu == "Insert":
    st.subheader("➕📝⌨️ Insert Data")
    insert_table = st.selectbox("Choose table to insert into", list(db_collections.keys()))

    if insert_table == "Countries and Regions":
        st.subheader("Add Country and Region")
        country = st.text_input("Country")
        region = st.text_input("Region")
        if st.button("Insert"):
            db.countries_and_regions.insert_one({"country": country, "region": region})
            st.success("Inserted successfully!")

    elif insert_table == "Climate":
        st.subheader("Add Climate Data")
        cr_str = st.selectbox("Country & Region", get_country_region_options())
        year = st.slider("Year", 1990, 2030, 2025)
        avg_temp = st.slider("Average Temperature (°C)", -10.0, 50.0, 20.0, step=0.01)
        precipitation = st.slider("Total Precipitation (mm)", 100.0, 3000.0, 500.0, step=0.01)
        co2 = st.slider("CO2 Emissions (MT)", 0.0, 50.0, 5.0, step=0.01)
        extreme = st.slider("Extreme Weather Events", 0, 20, 0)

        if st.button("Insert"):
            cr_id = get_country_region_id(cr_str)
            db.climate.insert_one({
                "year": year,
                "average_temperature_c": avg_temp,
                "total_precipitation_mm": precipitation,
                "co2_emissions_mt": co2,
                "extreme_weather_events": extreme,
                "country_region_id": cr_id
            })
            st.success("Inserted successfully!")

    elif insert_table == "Crops":
        st.subheader("Add Crop Data")
        cr_str = st.selectbox("Country & Region", get_country_region_options())

        crop_types = db.crops.distinct("crop_type")
        crop_type = st.selectbox("Crop Type", crop_types if crop_types else ["Wheat", "Corn", "Rice"])

        irrigation = st.slider("Irrigation Access (%)", 0.0, 100.0, 50.0, step=0.01)
        pesticide = st.slider("Pesticide Use (KG/HA)", 0.0, 100.0, 10.0, step=0.01)
        fertilizer = st.slider("Fertilizer Use (KG/HA)", 0.0, 200.0, 50.0, step=0.01)
        soil = st.slider("Soil Health Index", 20.0, 100.0, 60.0, step=0.01)
        strategy = st.text_input("Adaptation Strategies")

        climate_id = None
        if cr_str:
            cr_id = get_country_region_id(cr_str)

            # -------- CLIMATE REFERENCE --------
            climate_docs = list(db.climate.find({"country_region_id": cr_id}))
            climate_display_map = {}
            for c in climate_docs:
                display_str =", ".join([f"{k}: {v}" for k, v in c.items() if k != "_id"])
                climate_display_map[display_str] = c["_id"]

            if climate_display_map:
                climate_str = st.selectbox("Climate Reference", list(climate_display_map.keys()))
                climate_id = climate_display_map[climate_str]
            else:
                st.warning("No climate data available for this country/region.")

        if st.button("Insert") and climate_id:
            db.crops.insert_one({
                "crop_type": crop_type,
                "irrigation_access_percent": irrigation,
                "pesticide_use_kg_per_ha": pesticide,
                "fertilizer_use_kg_per_ha": fertilizer,
                "soil_health_index": soil,
                "adaptation_strategies": strategy,
                "country_region_id": cr_id,
                "climate_id": climate_id
            })
            st.success("Inserted successfully!")

    elif insert_table == "Productivity and Economic Impact":
        st.subheader("Add Productivity & Economic Data")
        cr_str = st.selectbox("Country & Region", get_country_region_options())
        yield_ = st.slider("Crop Yield (MT/HA)", min_value=0.0, max_value=10.0)
        econ = st.slider("Economic Impact (Million USD)", min_value=30.0, max_value=3000.0)

        climate_id = None
        crop_id = None

        if cr_str:
            cr_id = get_country_region_id(cr_str)

            # -------- CLIMATE REFERENCE --------
            climate_docs = list(db.climate.find({"country_region_id": cr_id}))
            climate_display_map = {}
            for c in climate_docs:
                display_str =", ".join([f"{k}: {v}" for k, v in c.items() if k != "_id"])
                climate_display_map[display_str] = c["_id"]

            if climate_display_map:
                climate_str = st.selectbox("Climate Reference", list(climate_display_map.keys()))
                climate_id = climate_display_map[climate_str]
            else:
                st.warning("No climate data available for this country/region.")

            # -------- CROP REFERNCE --------
            crop_docs = list(db.crops.find({"country_region_id": cr_id}))
            crop_display_map = {}
            for c in crop_docs:
                display_str =", ".join([f"{k}: {v}" for k, v in c.items() if k != "_id"])
                crop_display_map[display_str] = c["_id"]

            if crop_display_map:
                crop_str = st.selectbox("Crop Reference", list(crop_display_map.keys()))
                crop_id = crop_display_map[crop_str]
            else:
                st.warning("No crop data available for this country/region.")

        if st.button("Insert") and climate_id and crop_id:
            db.productivity_and_economic_impact.insert_one({
                "crop_yield_mt_per_ha": yield_,
                "economic_impact_million_usd": econ,
                "country_region_id": cr_id,
                "climate_id": climate_id,
                "crop_id": crop_id
            })
            st.success("Inserted successfully!")




# ------------------ SELECT -------------------
elif menu == "Select":
    st.subheader("🧐 Select Data")
    table_name = st.selectbox("Choose table to view", list(db_collections.keys()))
    collection = db_collections[table_name]
    
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    if df.empty:
        st.warning("No data found in this table.")
    else:
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

       
        num_rows = st.slider(
            "Select number of rows to display", 
            min_value=1, 
            max_value=len(df), 
            value=min(10, len(df))
        )
        
        st.dataframe(df.head(num_rows), use_container_width=True)



# ------------------ SEARCH -------------------
elif menu == "Search":
    st.subheader("🔍 Search Data")

    search_method = st.radio("Choose Search Method", [
        "Search by Country & Year",
        "Search by Country & Crop Type",
        "Search by Year & Crop Type",
        "climate impact on crop yield",
        "crop factor impact on crop yield",
        "HeatMap",
        "View Countries Producing a Specific Crop",
        "Fertilizer Usage Analysis",
        "Pesticide Usage Analysis",
        "Economic Impact Range",
        "Crop Yield Range",
        "TOP crop yield",
        "TOP economic impact"
    ])

    # ========== ✅ Search by Country & Year ==========
    if search_method == "Search by Country & Year":
        country_region_str = st.selectbox("Select Country & Region", get_country_region_options())

        if country_region_str:
            cr_id = get_country_region_id(country_region_str)

            climate_years = sorted({doc.get("year") for doc in db.climate.find({"country_region_id": cr_id}, {"year": 1}) if "year" in doc})

            if climate_years:
                selected_year = st.selectbox("Select Year", climate_years)

                st.markdown("### 🌡️ Climate Data")
                climate_data = list(db.climate.find({"country_region_id": cr_id, "year": selected_year}))
                if climate_data:
                    st.dataframe(pd.DataFrame(climate_data).drop(columns=["_id", "country_region_id"]), use_container_width=True)
                else:
                    st.info("No climate data found.")

                climate_ids = [doc["_id"] for doc in climate_data]

                st.markdown("### 🌾 Crop Data")
                crop_data = list(db.crops.find({"climate_id": {"$in": climate_ids}, "country_region_id": cr_id}))
                if crop_data:
                    st.dataframe(pd.DataFrame(crop_data).drop(columns=["_id", "country_region_id", "climate_id"]), use_container_width=True)
                else:
                    st.info("No crop data found for selected year and region.")


                st.markdown("### 📈 Productivity & Economic Impact")
                productivity_data_raw = list(db.productivity_and_economic_impact.find({
                    "climate_id": {"$in": climate_ids},
                    "country_region_id": cr_id
                }))
                
                productivity_data = []

                for prod in productivity_data_raw:
                    crop_id = prod.get("crop_id")
                    crop_doc = db.crops.find_one({"_id": crop_id}) if crop_id else None
                    prod["crop_name"] = crop_doc["crop_type"] if crop_doc else "Unknown"
                    prod.pop("_id", None)
                    prod.pop("country_region_id", None)
                    prod.pop("climate_id", None)
                    prod.pop("crop_id", None)
                    productivity_data.append(prod)

                if productivity_data:
                    st.dataframe(pd.DataFrame(productivity_data), use_container_width=True)
                    
                else:
                    st.info("No productivity data found.")
            else:
                st.warning("No climate years found for this country & region.")



              # ========== ✅ Search by Country & Crop Type ==========

    elif search_method == "Search by Country & Crop Type":
        country_region_str = st.selectbox("Select Country & Region", get_country_region_options())

        if country_region_str:
            cr_id = get_country_region_id(country_region_str)

            crops_cursor = db.crops.find({"country_region_id": cr_id})
            crop_options = sorted({doc["crop_type"] for doc in crops_cursor if "crop_type" in doc})

            if crop_options:
                selected_crop = st.selectbox("Select Crop Type", crop_options)

                selected_crops = list(db.crops.find({"country_region_id": cr_id, "crop_type": selected_crop}))

                if selected_crops:
                    crop_ids = [crop["_id"] for crop in selected_crops]

                    st.markdown("### 🌡️ Climate Data")
                    climate_ids = db.productivity_and_economic_impact.distinct("climate_id", {"crop_id": {"$in": crop_ids}})
                    climates = list(db.climate.find({"_id": {"$in": climate_ids}}))
                    if climates:
                        st.dataframe(pd.DataFrame(climates).drop(columns=["_id", "country_region_id"]), use_container_width=True)
                    else:
                        st.info("No climate data found.")

                    st.markdown("### 🌾 Crop Data")
                    st.dataframe(pd.DataFrame(selected_crops).drop(columns=["_id", "country_region_id"]), use_container_width=True)

                    st.markdown("### 📈 Productivity & Economic Impact")
                    productivity_docs = list(db.productivity_and_economic_impact.find({"crop_id": {"$in": crop_ids}}))

                    productivity_data = []
                    for prod in productivity_docs:
                        crop_id = prod.get("crop_id")
                        crop_doc = db.crops.find_one({"_id": crop_id})
                        prod["crop_name"] = crop_doc["crop_type"] if crop_doc else "Unknown"
                        prod.pop("_id", None)
                        prod.pop("country_region_id", None)
                        prod.pop("climate_id", None)
                        prod.pop("crop_id", None)
                        productivity_data.append(prod)

                    if productivity_data:
                        st.dataframe(pd.DataFrame(productivity_data), use_container_width=True)
                    else:
                        st.info("No productivity data found.")
                else:
                    st.warning("No crops found with the selected crop type.")

    elif search_method == "Search by Year & Crop Type":
        years = sorted({doc.get("year") for doc in db.climate.find({}, {"year": 1}) if "year" in doc})
        crops_available = sorted({doc.get("crop_type") for doc in db.crops.find({}, {"crop_type": 1}) if "crop_type" in doc})

        selected_year = st.selectbox("Select Year", years)
        selected_crop_type = st.selectbox("Select Crop Type", crops_available)

        matching_crops = list(db.crops.find({"crop_type": selected_crop_type}))
        crop_ids = [c["_id"] for c in matching_crops]

        matching_productivity = list(db.productivity_and_economic_impact.find({
            "crop_id": {"$in": crop_ids}
        }))

        filtered_productivity = []
        for prod in matching_productivity:
            climate_doc = db.climate.find_one({"_id": prod["climate_id"]})
            if climate_doc and climate_doc.get("year") == selected_year:
                prod["climate"] = climate_doc
                filtered_productivity.append(prod)

        if filtered_productivity:
            st.markdown("### 🌡️ Climate Data")
            climate_data_with_region = []
            for prod in filtered_productivity:
                climate_doc = prod["climate"]
                country_region = db.countries_and_regions.find_one({"_id": climate_doc["country_region_id"]})
                country_region_str = f"{country_region['country']} - {country_region['region']}" if country_region else "Unknown"
                climate_entry = {
                    "Country & Region": country_region_str,
                    "Year": climate_doc["year"],
                    "Avg Temp (°C)": climate_doc["average_temperature_c"],
                    "Total Precip. (mm)": climate_doc["total_precipitation_mm"],
                    "CO2 Emissions (MT)": climate_doc["co2_emissions_mt"],
                    "Extreme Events": climate_doc["extreme_weather_events"]
                }
                climate_data_with_region.append(climate_entry)

            st.dataframe(pd.DataFrame(climate_data_with_region), use_container_width=True)

            st.markdown("### 🌾 Crop Data")
            crop_data = []
            for prod in filtered_productivity:
                crop_doc = db.crops.find_one({"_id": prod["crop_id"]})
                if crop_doc:
                    crop_data.append({
                        "Crop Type": crop_doc["crop_type"],
                        "Irrigation Access (%)": crop_doc["irrigation_access_percent"],
                        "Pesticide Use (kg/ha)": crop_doc["pesticide_use_kg_per_ha"],
                        "Fertilizer Use (kg/ha)": crop_doc["fertilizer_use_kg_per_ha"],
                        "Soil Health Index": crop_doc["soil_health_index"],
                        "Adaptation Strategies": crop_doc["adaptation_strategies"]
                    })
            st.dataframe(pd.DataFrame(crop_data), use_container_width=True)

            st.markdown("### 📈 Productivity & Economic Impact")
            productivity_data = []
            for prod in filtered_productivity:
                prod_entry = {
                    "Crop Yield (MT/ha)": prod["crop_yield_mt_per_ha"],
                    "Economic Impact (M USD)": prod["economic_impact_million_usd"]
                }
                productivity_data.append(prod_entry)

            st.dataframe(pd.DataFrame(productivity_data), use_container_width=True)
        else:
            st.info("No data found for the selected year and crop type.")



    elif search_method == "climate impact on crop yield": 
        st.subheader("🌡️ Climate Factor Impact on Crop Yield")

        country_region_names = {
            str(doc["_id"]): f"{doc['country']} - {doc['region']}"
            for doc in db_collections["Countries and Regions"].find()
        }

        selected_cr_id = st.selectbox("Choose Country & Region", list(country_region_names.keys()), format_func=lambda x: country_region_names[x])

        climate_factors = {
            "Average Temperature (°C)": "average_temperature_c",
            "Total Precipitation (mm)": "total_precipitation_mm",
            "CO2 Emissions (Mt)": "co2_emissions_mt",
            "Extreme Weather Events": "extreme_weather_events"
        }

        selected_factor_label = st.selectbox("Choose Climate Factor", list(climate_factors.keys()))
        selected_factor = climate_factors[selected_factor_label]


        climate_docs = list(db_collections["Climate"].find({"country_region_id": ObjectId(selected_cr_id)}))
        climate_df = pd.DataFrame(climate_docs)

        if not climate_df.empty:
            climate_df = climate_df[["_id", "year", selected_factor]]
            climate_df = climate_df.dropna(subset=["year", selected_factor])
            climate_df.rename(columns={"_id": "climate_id"}, inplace=True)

            impact_docs = list(db_collections["Productivity and Economic Impact"].find({"country_region_id": ObjectId(selected_cr_id)}))
            impact_df = pd.DataFrame(impact_docs)

            if not impact_df.empty:
                impact_df = impact_df[["climate_id", "crop_yield_mt_per_ha"]]
                impact_df = impact_df.dropna(subset=["crop_yield_mt_per_ha"])

            
                merged_df = pd.merge(climate_df, impact_df, on="climate_id")

                if not merged_df.empty:
                    grouped = merged_df.groupby("year").agg({
                        selected_factor: "mean",
                        "crop_yield_mt_per_ha": "mean"
                    }).reset_index()

                    st.write("### Yearly Averages")
                    st.dataframe(grouped)

                    fig = px.line(
                        grouped,
                        x="year",
                        y=[selected_factor, "crop_yield_mt_per_ha"],
                        labels={
                            "value": "Value",
                            "variable": "Metric",
                            "year": "Year"
                        },
                        title=f"{selected_factor_label} vs Crop Yield Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    correlation = grouped[selected_factor].corr(grouped["crop_yield_mt_per_ha"])
                    st.write(f"**📊 Correlation between `{selected_factor_label}` and Crop Yield:** `{correlation:.3f}`")
                else:
                    st.warning("No merged data available for the selected region.")
            else:
                st.warning("No productivity and economic impact data found for this region.")
        else:
             st.warning("No climate data found for this region.")



    elif search_method == "crop factor impact on crop yield": 
        st.subheader("🌾 Crop Factor Impact on Crop Yield")

        country_region_names = {
            str(doc["_id"]): f"{doc['country']} - {doc['region']}"
            for doc in db_collections["Countries and Regions"].find()
        }

        selected_cr_id = st.selectbox("Choose Country & Region", list(country_region_names.keys()), format_func=lambda x: country_region_names[x])

        crop_factors = {
            "Irrigation Access (%)": "irrigation_access_percent",
            "Pesticide Use (kg/ha)": "pesticide_use_kg_per_ha",
            "Fertilizer Use (kg/ha)": "fertilizer_use_kg_per_ha",
            "Soil Health Index": "soil_health_index"
        }

        selected_crop_label = st.selectbox("Choose Crop Factor", list(crop_factors.keys()))
        selected_crop_factor = crop_factors[selected_crop_label]

        crops_docs = list(db_collections["Crops"].find({"country_region_id": ObjectId(selected_cr_id)}))
        crops_df = pd.DataFrame(crops_docs)

        if not crops_df.empty:
            crops_df = crops_df[["_id", selected_crop_factor]]
            crops_df.rename(columns={"_id": "crop_id"}, inplace=True)
            crops_df = crops_df.dropna(subset=[selected_crop_factor])

            impact_docs = list(db_collections["Productivity and Economic Impact"].find({"country_region_id": ObjectId(selected_cr_id)}))
            impact_df = pd.DataFrame(impact_docs)

            if not impact_df.empty:
                impact_df = impact_df[["crop_id", "crop_yield_mt_per_ha"]]
                impact_df = impact_df.dropna(subset=["crop_yield_mt_per_ha"])

                merged_df = pd.merge(crops_df, impact_df, on="crop_id")

                if not merged_df.empty:
                    st.write("### Data Summary")
                    st.dataframe(merged_df)

                    fig = px.scatter(
                        merged_df,
                        x=selected_crop_factor,
                        y="crop_yield_mt_per_ha",
                        trendline="ols",
                        labels={
                            selected_crop_factor: selected_crop_label,
                            "crop_yield_mt_per_ha": "Crop Yield (mt/ha)"
                        },
                        title=f"{selected_crop_label} vs Crop Yield"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    correlation = merged_df[selected_crop_factor].corr(merged_df["crop_yield_mt_per_ha"])
                    st.write(f"**📊 Correlation between `{selected_crop_label}` and Crop Yield:** `{correlation:.3f}`")
                else:
                    st.warning("No merged data available for the selected region.")
            else:
                st.warning("No productivity and economic impact data found for this region.")
        else:
            st.warning("No crop data found for this region.")


    elif search_method == "View Countries Producing a Specific Crop":
        st.subheader("Select a Crop to View Producing Countries")

        crop_types = sorted({c["crop_type"] for c in db.crops.find({}, {"crop_type": 1}) if "crop_type" in c})

        selected_crop = st.selectbox("Choose a crop type", crop_types)

        if selected_crop:
            matched_crops = list(db.crops.find({"crop_type": selected_crop}))

            region_ids = list({crop["country_region_id"] for crop in matched_crops if "country_region_id" in crop})

            countries = list(db.countries_and_regions.find({"_id": {"$in": region_ids}}))

            if countries:
                df = pd.DataFrame(countries)
                df = df.rename(columns={"country": "Country", "region": "Region"})
                st.dataframe(df[["Country", "Region"]])
            else:
                st.info("No countries found for the selected crop.")

    elif search_method == "Fertilizer Usage Analysis":
        st.subheader("🔍 Find Years and Regions with High Fertilizer Use")

        threshold = st.slider("Minimum fertilizer use (kg/ha)", 0, 100, 50)

        matching_docs = list(db.crops.find({"fertilizer_use_kg_per_ha": {"$gt": threshold}}))

        results = []

        for crop in matching_docs:
            cr_id = crop.get("country_region_id")
            climate_id = crop.get("climate_id")

            cr_doc = db.countries_and_regions.find_one({"_id": cr_id}) if cr_id else None
            climate_doc = db.climate.find_one({"_id": climate_id}) if climate_id else None

            if cr_doc and climate_doc:
                results.append({
                    "Year": climate_doc.get("year"),
                    "Country": cr_doc.get("country"),
                    "Region": cr_doc.get("region"),
                    "Fertilizer Use (kg/ha)": crop.get("fertilizer_use_kg_per_ha")
                })

        if results:
            st.markdown(f"### 📋 Regions with Fertilizer Use > {threshold} kg/ha")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("No matching records found.")



    elif search_method == "Pesticide Usage Analysis":
        st.subheader("🔍 Find Years and Regions with High Pesticide Use")

        pesticide_threshold = st.slider("Minimum pesticide use (kg/ha)", 0, 50, 10)

        matching_docs = list(db.crops.find({"pesticide_use_kg_per_ha": {"$gt": pesticide_threshold}}))

        results = []

        for crop in matching_docs:
            cr_id = crop.get("country_region_id")
            climate_id = crop.get("climate_id")

            cr_doc = db.countries_and_regions.find_one({"_id": cr_id}) if cr_id else None
            climate_doc = db.climate.find_one({"_id": climate_id}) if climate_id else None

            if cr_doc and climate_doc:
                results.append({
                    "Year": climate_doc.get("year"),
                    "Country": cr_doc.get("country"),
                    "Region": cr_doc.get("region"),
                    "Pesticide Use (kg/ha)": crop.get("pesticide_use_kg_per_ha")
                })

        if results:
            st.markdown(f"### 📋 Regions with Pesticide Use > {pesticide_threshold} kg/ha")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("No matching records found.")


    elif search_method == "Economic Impact Range": 
        countries_collection = db["countries_and_regions"]
        climate_collection = db["climate"]
        crops_collection = db["crops"]
        productivity_collection = db["productivity_and_economic_impact"]

        df_productivity = pd.DataFrame(list(productivity_collection.find()))
        df_climate = pd.DataFrame(list(climate_collection.find()))
        df_crops = pd.DataFrame(list(crops_collection.find()))
        df_countries = pd.DataFrame(list(countries_collection.find()))

        df_productivity["country_region_id"] = df_productivity["country_region_id"].astype(str)
        df_productivity["climate_id"] = df_productivity["climate_id"].astype(str)
        df_productivity["crop_id"] = df_productivity["crop_id"].astype(str)
        df_climate["_id"] = df_climate["_id"].astype(str)
        df_crops["_id"] = df_crops["_id"].astype(str)
        df_countries["_id"] = df_countries["_id"].astype(str)

        min_impact = float(df_productivity["economic_impact_million_usd"].min())
        max_impact = float(df_productivity["economic_impact_million_usd"].max())

        impact_range = st.slider(
            "Choose range",
            min_value=round(min_impact, 2),
            max_value=round(max_impact, 2),
            value=(round(min_impact, 2), round(max_impact, 2)),
            step=1.0
        )

        filtered_productivity = df_productivity[
            (df_productivity["economic_impact_million_usd"] >= impact_range[0]) &
            (df_productivity["economic_impact_million_usd"] <= impact_range[1])
        ]

        st.subheader("📊 Productivity and Economic Impact")
        if filtered_productivity.empty:
            st.warning("No results")
        else:
            display_df = filtered_productivity.drop(columns=["_id"]).reset_index(drop=True)
            display_df.index += 1
            st.dataframe(display_df)

            related_country_ids = filtered_productivity["country_region_id"].unique().tolist()
            related_climate_ids = filtered_productivity["climate_id"].unique().tolist()
            related_crop_ids = filtered_productivity["crop_id"].unique().tolist()

            filtered_countries = df_countries[df_countries["_id"].isin(related_country_ids)]
            st.subheader("🗺️ Countries and Regions related")
            display_countries = filtered_countries.drop(columns=["_id"]).reset_index(drop=True)
            display_countries.index += 1
            st.dataframe(display_countries)

            filtered_climate = df_climate[df_climate["_id"].isin(related_climate_ids)]
            st.subheader("🌦️ Climate relted")
            display_climate = filtered_climate.drop(columns=["_id"]).reset_index(drop=True)
            display_climate.index += 1
            st.dataframe(display_climate)

        # جدول Crops
            filtered_crops = df_crops[df_crops["_id"].isin(related_crop_ids)]
            st.subheader("🌾 Crops related")
            display_crops = filtered_crops.drop(columns=["_id"]).reset_index(drop=True)
            display_crops.index += 1
            st.dataframe(display_crops)

    elif search_method == "Crop Yield Range":
        countries_collection = db["countries_and_regions"]
        climate_collection = db["climate"]
        crops_collection = db["crops"]
        productivity_collection = db["productivity_and_economic_impact"]

        df_productivity = pd.DataFrame(list(productivity_collection.find()))
        df_climate = pd.DataFrame(list(climate_collection.find()))
        df_crops = pd.DataFrame(list(crops_collection.find()))
        df_countries = pd.DataFrame(list(countries_collection.find()))

        df_productivity["country_region_id"] = df_productivity["country_region_id"].astype(str)
        df_productivity["climate_id"] = df_productivity["climate_id"].astype(str)
        df_productivity["crop_id"] = df_productivity["crop_id"].astype(str)
        df_climate["_id"] = df_climate["_id"].astype(str)
        df_crops["_id"] = df_crops["_id"].astype(str)
        df_countries["_id"] = df_countries["_id"].astype(str)

        min_yield = float(df_productivity["crop_yield_mt_per_ha"].min())
        max_yield = float(df_productivity["crop_yield_mt_per_ha"].max())

        yield_range = st.slider(
            "Choose range",
            min_value=round(min_yield, 2),
            max_value=round(max_yield, 2),
            value=(round(min_yield, 2), round(max_yield, 2)),
            step=0.01
        )

        filtered_productivity = df_productivity[
            (df_productivity["crop_yield_mt_per_ha"] >= yield_range[0]) &
            (df_productivity["crop_yield_mt_per_ha"] <= yield_range[1])
        ]

        if filtered_productivity.empty:
            st.warning("No results")
        else:
            display_df = filtered_productivity.drop(columns=["_id"]).reset_index(drop=True)
            display_df.index += 1
            st.dataframe(display_df)

            related_country_ids = filtered_productivity["country_region_id"].unique().tolist()
            related_climate_ids = filtered_productivity["climate_id"].unique().tolist()
            related_crop_ids = filtered_productivity["crop_id"].unique().tolist()

            filtered_countries = df_countries[df_countries["_id"].isin(related_country_ids)]
            st.subheader("🗺️ Countries and Regions related")
            display_countries = filtered_countries.drop(columns=["_id"]).reset_index(drop=True)
            display_countries.index += 1
            st.dataframe(display_countries)

            filtered_climate = df_climate[df_climate["_id"].isin(related_climate_ids)]
            st.subheader("🌦️ Climate related")
            display_climate = filtered_climate.drop(columns=["_id"]).reset_index(drop=True)
            display_climate.index += 1
            st.dataframe(display_climate)

            filtered_crops = df_crops[df_crops["_id"].isin(related_crop_ids)]
            st.subheader("🌽 Crops related")
            display_crops = filtered_crops.drop(columns=["_id"]).reset_index(drop=True)
            display_crops.index += 1
            st.dataframe(display_crops)

    elif search_method == "TOP crop yield":
        st.subheader("🌾 Top Crop Yield Records")

        top_n = st.slider("Select number of top records to display", min_value=1, max_value=20, value=5)

        top_yields = list(
            db.productivity_and_economic_impact.find()
            .sort("crop_yield_mt_per_ha", -1)
            .limit(top_n)
        )

        results = []
        for record in top_yields:
            combined = record.copy()

            country_region_id = record.get("country_region_id")
            if country_region_id:
                country_data = db.countries_and_regions.find_one({"_id": country_region_id})
                if country_data:
                    combined.update({
                        "country": country_data.get("country"),
                        "region": country_data.get("region"),
                    })

            climate_id = record.get("climate_id")
            if climate_id:
                climate_data = db.climate.find_one({"_id": climate_id})
                if climate_data:
                    combined.update({
                        "year": climate_data.get("year"),
                        "average_temperature_c": climate_data.get("average_temperature_c"),
                        "total_precipitation_mm": climate_data.get("total_precipitation_mm"),
                        "co2_emissions_mt": climate_data.get("co2_emissions_mt"),
                        "extreme_weather_events": climate_data.get("extreme_weather_events"),
                    })

            crop_data = db.crops.find_one({"country_region_id": country_region_id,"climate_id": climate_id})
            if crop_data:
                combined.update({
                    "crop_type": crop_data.get("crop_type"),
                    "irrigation_access_percent": crop_data.get("irrigation_access_percent"),
                    "pesticide_use_kg_per_ha": crop_data.get("pesticide_use_kg_per_ha"),
                    "fertilizer_use_kg_per_ha": crop_data.get("fertilizer_use_kg_per_ha"),
                    "soil_health_index": crop_data.get("soil_health_index"),
                    "adaptation_strategies": crop_data.get("adaptation_strategies"),
                })

            results.append(combined)

        if results:
            df = pd.DataFrame(results)
            df.drop(columns=["_id", "country_region_id", "climate_id","crop_id"], errors="ignore", inplace=True)
            st.dataframe(df)
        else:
            st.info("No data found.")


    elif search_method == "TOP economic impact":
        st.subheader("💰 Top Economic Impact Records")

        top_n = st.slider("Select number of top records to display", min_value=1, max_value=20, value=5, key="economic_slider")

        top_impacts = list(
            db.productivity_and_economic_impact.find()
            .sort("economic_impact_million_usd", -1)
            .limit(top_n)
        )

        results = []
        for record in top_impacts:
            combined = record.copy()

            country_region_id = record.get("country_region_id")
            if country_region_id:
                country_data = db.countries_and_regions.find_one({"_id": country_region_id})
                if country_data:
                    combined.update({
                        "country": country_data.get("country"),
                        "region": country_data.get("region"),
                    })

            climate_id = record.get("climate_id")
            if climate_id:
                climate_data = db.climate.find_one({"_id": climate_id})
                if climate_data:
                    combined.update({
                        "year": climate_data.get("year"),
                        "average_temperature_c": climate_data.get("average_temperature_c"),
                        "total_precipitation_mm": climate_data.get("total_precipitation_mm"),
                        "co2_emissions_mt": climate_data.get("co2_emissions_mt"),
                        "extreme_weather_events": climate_data.get("extreme_weather_events"),
                    })

            crop_data = db.crops.find_one({"country_region_id": country_region_id,"climate_id": climate_id})
            if crop_data:
                combined.update({
                    "crop_type": crop_data.get("crop_type"),
                    "irrigation_access_percent": crop_data.get("irrigation_access_percent"),
                    "pesticide_use_kg_per_ha": crop_data.get("pesticide_use_kg_per_ha"),
                    "fertilizer_use_kg_per_ha": crop_data.get("fertilizer_use_kg_per_ha"),
                    "soil_health_index": crop_data.get("soil_health_index"),
                    "adaptation_strategies": crop_data.get("adaptation_strategies"),
                })

            results.append(combined)

        if results:
            df = pd.DataFrame(results)
            df.drop(columns=["_id", "country_region_id", "climate_id"], errors="ignore", inplace=True)
            st.dataframe(df)
        else:
            st.info("No data found.")

    elif search_method == "HeatMap":
        st.subheader("📊 Correlation Heatmap with Crop Yield")

        # قراءة بيانات Productivity and Economic Impact
        pei_data = list(db_collections["Productivity and Economic Impact"].find())

        if not pei_data:
            st.warning("No data available to generate heatmap.")
        else:
            # ربط بيانات Climate و Crops
            for doc in pei_data:
                # دمج بيانات Climate
                climate_doc = db.climate.find_one({"_id": doc.get("climate_id")})
                if climate_doc:
                    doc.update({
                        "average_temperature_c": climate_doc.get("average_temperature_c"),
                        "total_precipitation_mm": climate_doc.get("total_precipitation_mm"),
                        "co2_emissions_mt": climate_doc.get("co2_emissions_mt"),
                        "extreme_weather_events": climate_doc.get("extreme_weather_events"),
                    })

                # دمج بيانات Crops
                crop_doc = db.crops.find_one({"_id": doc.get("crop_id")})
                if crop_doc:
                    doc.update({
                        "irrigation_access_percent": crop_doc.get("irrigation_access_percent"),
                        "pesticide_use_kg_per_ha": crop_doc.get("pesticide_use_kg_per_ha"),
                        "fertilizer_use_kg_per_ha": crop_doc.get("fertilizer_use_kg_per_ha"),
                        "soil_health_index": crop_doc.get("soil_health_index"),
                    })

            # تحويل إلى DataFrame
            df = pd.DataFrame(pei_data)

            # اختيار المتغيرات الرقمية فقط
            numeric_df = df.select_dtypes(include=["int64", "float64"])

            if "crop_yield_mt_per_ha" not in numeric_df.columns:
                st.warning("Crop yield data not found.")
            else:
                # حساب الارتباط
                corr = numeric_df.corr()

                # رسم Heatmap
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr[["crop_yield_mt_per_ha"]].sort_values("crop_yield_mt_per_ha", ascending=False), 
                            annot=True, cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
    

elif menu == "Indexing":
    st.header("🔎 Indexing")

    def create_indexes_safely():
        index_specs = {
            "Countries and Regions": [("country", 1), ("region", 1)],
            "Climate": [("country_region_id", 1), ("extreme_weather_events", 1), ("year", 1)],
            "Crops": [("country_region_id", 1), ("climate_id", 1)],
            "Productivity and Economic Impact": [("country_region_id", 1), ("climate_id", 1), ("crop_id", 1)],
        }

        for name, fields in index_specs.items():
            collection = db_collections[name]
            index_name = "_".join([f"{f[0]}_{f[1]}" for f in fields])
            existing_indexes = collection.index_information()
            if index_name not in existing_indexes:
                collection.create_index(fields, name=index_name)

    if "indexes_created" not in st.session_state:
        create_indexes_safely()
        st.session_state.indexes_created = True

    search_option = st.radio("Select Search method", ["Country & Region", "Extreme Weather Events", "Year"])

    countries_collection = db_collections["Countries and Regions"]
    climate_collection = db_collections["Climate"]
    crops_collection = db_collections["Crops"]
    productivity_collection = db_collections["Productivity and Economic Impact"]

    if search_option == "Country & Region":
        country_region_docs = list(countries_collection.find())
        country_region_options = [(doc['country'], doc['region']) for doc in country_region_docs]

        selected = st.selectbox("Select country & region", sorted(country_region_options))
        selected_country, selected_region = selected

        country_region_doc = countries_collection.find_one({'country': selected_country, 'region': selected_region})

        if not country_region_doc:
            st.warning("Not found country-region")
        else:
            cr_id = country_region_doc['_id']
            climate_docs = list(climate_collection.find({'country_region_id': cr_id}))
            combined_rows = []

            for climate in climate_docs:
                climate_id = climate['_id']
                crop_docs = list(crops_collection.find({'country_region_id': cr_id, 'climate_id': climate_id}))
                
                for crop in crop_docs:
                    crop_id = crop['_id']
                    prod_docs = list(productivity_collection.find({
                        'country_region_id': cr_id,
                        'climate_id': climate_id,
                        'crop_id': crop_id
                    }))
                    for prod in prod_docs:
                        row = {
                            'Country': selected_country,
                            'Region': selected_region,
                            'Year': climate.get('year'),
                            'Avg Temp (C)': climate.get('average_temperature_c'),
                            'Precipitation (mm)': climate.get('total_precipitation_mm'),
                            'CO2 Emissions (MT)': climate.get('co2_emissions_mt'),
                            'Extreme Events': climate.get('extreme_weather_events'),
                            'Crop Type': crop.get('crop_type'),
                            'Irrigation %': crop.get('irrigation_access_percent'),
                            'Pesticide Use': crop.get('pesticide_use_kg_per_ha'),
                            'Fertilizer Use': crop.get('fertilizer_use_kg_per_ha'),
                            'Soil Health': crop.get('soil_health_index'),
                            'Yield (MT/ha)': prod.get('crop_yield_mt_per_ha'),
                            'Economic Impact (M USD)': prod.get('economic_impact_million_usd')
                        }
                        combined_rows.append(row)

            if combined_rows:
                st.dataframe(pd.DataFrame(combined_rows))
            else:
                st.info("Not found country & region")

    elif search_option == "Extreme Weather Events":
        st.subheader("🔍 Extreme Weather Events")
        selected_events = st.slider("Extreme Weather Events", min_value=0, max_value=10, step=1, value=3)

        climate_docs = list(climate_collection.find({'extreme_weather_events': selected_events}))
        combined_rows = []

        for climate in climate_docs:
            cr_id = climate['country_region_id']
            climate_id = climate['_id']

            country_doc = countries_collection.find_one({'_id': cr_id})
            if not country_doc:
                continue

            crop_docs = list(crops_collection.find({'country_region_id': cr_id, 'climate_id': climate_id}))
            for crop in crop_docs:
                crop_id = crop['_id']
                prod_docs = list(productivity_collection.find({
                    'country_region_id': cr_id,
                    'climate_id': climate_id,
                    'crop_id': crop_id
                }))
                for prod in prod_docs:
                    row = {
                        'Country & Region': f"{country_doc['country']} - {country_doc['region']}",
                        'Year': climate.get('year'),
                        'Extreme Events': climate.get('extreme_weather_events'),
                        'Yield (MT/ha)': prod.get('crop_yield_mt_per_ha'),
                        'Economic Impact (M USD)': prod.get('economic_impact_million_usd')
                    }
                    combined_rows.append(row)

        if combined_rows:
            df_combined = pd.DataFrame(combined_rows)
            st.dataframe(df_combined)
        else:
            st.info("Not found for this number of Extreme weather events")

    elif search_option == "Year":
        st.subheader("📅 Search by Year")
        all_years = sorted(climate_collection.distinct("year"))
        selected_year = st.selectbox("Select Year", all_years)

        climate_docs = list(climate_collection.find({"year": selected_year}))
        combined_rows = []

        for climate in climate_docs:
            cr_id = climate["country_region_id"]
            climate_id = climate["_id"]
            country_doc = countries_collection.find_one({"_id": cr_id})
            if not country_doc:
                continue

            crop_docs = list(crops_collection.find({"country_region_id": cr_id, "climate_id": climate_id}))
            for crop in crop_docs:
                crop_id = crop['_id']
                prod_docs = list(productivity_collection.find({
                    'country_region_id': cr_id,
                    'climate_id': climate_id,
                    'crop_id': crop_id
                }))
                for prod in prod_docs:
                    row = {
                        "Country": country_doc.get("country"),
                        "Region": country_doc.get("region"),
                        "Year": selected_year,
                        "Crop": crop.get("crop_type"),
                        "Avg Temp (C)": climate.get("average_temperature_c"),
                        "Total Precipitation (mm)": climate.get("total_precipitation_mm"),
                        "CO2 Emissions (MT)": climate.get("co2_emissions_mt"),
                        "Extreme Events": climate.get("extreme_weather_events"),
                        "Irrigation Access (%)": crop.get("irrigation_access_percent"),
                        "Pesticide Use (kg/ha)": crop.get("pesticide_use_kg_per_ha"),
                        "Fertilizer Use (kg/ha)": crop.get("fertilizer_use_kg_per_ha"),
                        "Soil Health Index": crop.get("soil_health_index"),
                        "Adaptation Strategies": crop.get("adaptation_strategies"),
                        "Yield (MT/ha)": prod.get("crop_yield_mt_per_ha"),
                         "Economic Impact (M USD)": prod.get("economic_impact_million_usd")
                    }
                    combined_rows.append(row)

        if combined_rows:
            df = pd.DataFrame(combined_rows)
            st.dataframe(df)
        else:
            st.info("No data for this year")






# -------------------------------
# 📦 Aggregation Functions
# -------------------------------

elif menu == "Aggregation":
    # def agg_avg_temperature_by_country():
        
    #     pipeline = [
    #     {
    #         "$lookup": {
    #             "from": "countries_and_regions",
    #             "localField": "country_region_id",
    #             "foreignField": "_id",
    #             "as": "region_info"
    #         }
    #     },
    #     {
    #         "$unwind": "$region_info"
    #     },
    #     {
    #         "$group": {
    #             "_id": "$region_info.country",
    #             "Avg Temperature (°C)": {"$avg": "$average_temperature_c"}
    #         }
    #     },
    #     {
    #         "$sort": {"Avg Temperature (°C)": -1}
    #     }
    # ]
    #     return list(db.climate.aggregate(pipeline))
    
    def agg_avg_temperature_by_country():
        pipeline = [
        {
            "$lookup": {
                "from": "countries_and_regions",
                "localField": "country_region_id",
                "foreignField": "_id",
                "as": "region_info"
            }
        },
        {"$unwind": "$region_info"},
        {
            "$group": {
                "_id": "$region_info.country",
                "Avg Temperature (°C)": {"$avg": "$average_temperature_c"}
            }
        },
        {"$sort": {"Avg Temperature (°C)": -1}}
]
        return list(db.climate.aggregate(pipeline))


    def agg_total_co2_by_year():
        pipeline = [
            {"$match": {"co2_emissions_mt": {"$ne": None}}},
            {"$group": {
                "_id": "$year",
                "TotalCO2": {"$sum": "$co2_emissions_mt"}
            }},
            {"$sort": {"_id": 1}},
            {"$project": {
                "Year": "$_id",
                "Total CO2 Emissions (MT)": "$TotalCO2",
                "_id": 0
            }}
        ]
        return list(db.climate.aggregate(pipeline))
    
    def agg_total_co2_by_country():
        pipeline = [
            {
                "$lookup": {
                    "from": "countries_and_regions",
                    "localField": "country_region_id",
                    "foreignField": "_id",
                    "as": "region_info"
                }
            },
            {"$unwind": "$region_info"},
            {
                "$group": {
                    "_id": "$region_info.country",
                    "Total CO2 Emissions (Mt)": {"$sum": "$co2_emissions_mt"}
                }
            },
            {"$sort": {"Total CO2 Emissions (Mt)": -1}}
        ]
        return list(db.climate.aggregate(pipeline))


    # def agg_total_yield_by_crop():
    #     pipeline = [
    #         {"$match": {"crop_yield_mt_per_ha": {"$ne": None}}},
    #         {"$group": {
    #             "_id": "$crop_type",
    #             "TotalYield": {"$sum": "$crop_yield_mt_per_ha"}
    #         }},
    #         {"$sort": {"TotalYield": -1}},
    #         {"$project": {
    #             "Crop Type": "$_id",
    #             "Total Yield (MT/ha)": "$TotalYield",
    #             "_id": 0
    #         }}
    #     ]
    #     return list(db.productivity_and_economic_impact.aggregate(pipeline))
    
    def agg_total_yield_by_crop():
        pipeline = [
            {
                "$lookup": {
                    "from": "crops",
                    "localField": "crop_id",
                    "foreignField": "_id",
                    "as": "crop_info"
                }
            },
            {"$unwind": "$crop_info"},
            {
                "$group": {
                    "_id": "$crop_info.crop_type",
                    "Total Yield (mt/ha)": {"$sum": "$crop_yield_mt_per_ha"}
                }
            },
            {"$sort": {"Total Yield (mt/ha)": -1}}
        ]
        return list(db.productivity_and_economic_impact.aggregate(pipeline))


    # def agg_avg_soil_health_by_crop_type():
    #     pipeline = [
    #         {"$match": {"soil_health_index": {"$ne": None}}},
    #         {"$group": {
    #             "_id": "$crop_type",
    #             "AvgSoilHealth": {"$avg": "$soil_health_index"}
    #         }},
    #         {"$sort": {"AvgSoilHealth": -1}},
    #         {"$project": {
    #             "Crop Type": "$_id",
    #             "Avg Soil Health Index": "$AvgSoilHealth",
    #             "_id": 0
    #         }}
    #     ]
    #     return list(db.crops.aggregate(pipeline))
    
    def agg_avg_soil_health_by_crop_type():
        pipeline = [
            {
                "$group": {
                    "_id": "$crop_type",
                    "Avg Soil Health Index": {"$avg": "$soil_health_index"}
                }
            },
            {"$sort": {"Avg Soil Health Index": -1}}
        ]
        return list(db.crops.aggregate(pipeline))
    
    
    def agg_economic_impact_by_country():
        pipeline = [
            {
                "$lookup": {
                    "from": "countries_and_regions",
                    "localField": "country_region_id",
                    "foreignField": "_id",
                    "as": "region_info"
                }
            },
            {"$unwind": "$region_info"},
            {
                "$group": {
                    "_id": "$region_info.country",
                    "Total Economic Impact (Million USD)": {"$sum": "$economic_impact_million_usd"}
                }
            },
            {"$sort": {"Total Economic Impact (Million USD)": -1}}
        ]
        return list(db.productivity_and_economic_impact.aggregate(pipeline))


    # -------------------------------
    # 🚀 Streamlit UI
    # -------------------------------'

    st.title("🌍 Climate & Agriculture Analytics (Aggregation Pipelines)")

    st.title("Select Analysis")
    menu = st.radio("Choose", [
        "Average Temperature by Country",
        "Total CO2 Emissions by Year",
        "Total CO2 Emissions by Country",
        "Total Yield by Crop",
        "Average Soil Health by Crop",
        "Economic Impact by country",
        "Top crop yield in country",
        "Top Economic Impact in country",
        "Top crop yield in year",
        "Top Economic Impact in Year"
    ])
    

    if menu == "Average Temperature by Country":
        results = agg_avg_temperature_by_country()
        df = pd.DataFrame(results)
        st.subheader("Average Temperature (°C) by Country")
        st.dataframe(df)

    elif menu == "Total CO2 Emissions by Year":
        results = agg_total_co2_by_year()
        df = pd.DataFrame(results)
        st.subheader("Total CO2 Emissions (MT) by Year")
        st.dataframe(df)
        
    elif menu == "Total CO2 Emissions by Country":
        results = agg_total_co2_by_country()
        df = pd.DataFrame(results)
        st.subheader("Total CO2 Emissions (MT) by Country")
        st.dataframe(df)

    elif menu == "Total Yield by Crop":
        results = agg_total_yield_by_crop()
        df = pd.DataFrame(results)
        st.subheader("Total Yield (MT/ha) by Crop Type")
        st.dataframe(df)

    elif menu == "Average Soil Health by Crop":
        results = agg_avg_soil_health_by_crop_type()
        df = pd.DataFrame(results)
        st.subheader("Average Soil Health Index by Crop Type")
        st.dataframe(df)


    elif menu == "Economic Impact by country":
        results = agg_economic_impact_by_country()
        df = pd.DataFrame(results)
        st.subheader("Economic Impact (Million USD) by country")
        st.dataframe(df)


    elif menu == "Top crop yield in country":
        st.subheader("Top Crop Yield Results")

        countries = list(db.countries_and_regions.find())
        country_names = [country["country"] for country in countries]
        selected_country = st.selectbox("Select Country", country_names)

        
        num_top_crops = st.slider("Select the number of top crops", 1, 10, 5)

        
        pipeline = [
            {
                "$lookup": {
                    "from": "crops",  # ربط جدول المحاصيل
                    "localField": "crop_id",  # الحقل في productivity_and_economic_impact الذي يربط مع المحاصيل
                    "foreignField": "_id",
                    "as": "crop_data"
                }
            },
            {
                "$unwind": "$crop_data"  # لفك الربط بين الـ crops
            },
            {
                "$lookup": {
                    "from": "climate",  # ربط جدول المناخ
                    "localField": "climate_id",  # الحقل في productivity_and_economic_impact الذي يربط مع المناخ
                    "foreignField": "_id",
                    "as": "climate_data"
                }
            },
            {
                "$unwind": "$climate_data"  # لفك الربط بين الـ climate
            },
            {
                "$lookup": {
                    "from": "countries_and_regions",  # ربط جدول البلدان
                    "localField": "country_region_id",  # الحقل في productivity_and_economic_impact الذي يربط مع countries_and_regions
                    "foreignField": "_id",
                    "as": "country_data"
                }
            },
            {
                "$unwind": "$country_data"  # لفك الربط بين الـ countries_and_regions
            },
            {
                "$match": {
                    "country_data.country": selected_country  # تصفية حسب الدولة المختارة
                }
            },
            {
                "$sort": {
                    "crop_yield_mt_per_ha": -1  # ترتيب حسب crop_yield_mt_per_ha
                }
            },
            {
                "$limit": num_top_crops  # أخذ أول N نتيجة حسب الـ slider
            }
        ]
        
       
        top_crops_data = list(db.productivity_and_economic_impact.aggregate(pipeline))

        if top_crops_data:
            
            results = []
            for data in top_crops_data:
                crop_data = data.get("crop_data", {})
                climate_data = data.get("climate_data", {})
                country_data = data.get("country_data", {})
                
                results.append({
                    "Crop": crop_data.get("crop_type", "N/A"),
                    "Yield (MT/ha)": data.get("crop_yield_mt_per_ha", "N/A"),
                    "Economic Impact (Million USD)": data.get("economic_impact_million_usd", "N/A"),
                    "Year": climate_data.get("year", "N/A"),
                    "Temperature (°C)": climate_data.get("average_temperature_c", "N/A"),
                    "Country": country_data.get("country", "N/A"),
                    "Region": country_data.get("region", "N/A"),
                    "Precipitation (mm)": climate_data.get("total_precipitation_mm", "N/A"),
                    "CO2 Emissions (MT)": climate_data.get("co2_emissions_mt", "N/A"),
                    "Extreme Weather Events": climate_data.get("extreme_weather_events", "N/A"),
                    "Irrigation Access (%)": crop_data.get("irrigation_access_percent", "N/A"),
                    "Pesticide Use (kg/ha)": crop_data.get("pesticide_use_kg_per_ha", "N/A"),
                    "Fertilizer Use (kg/ha)": crop_data.get("fertilizer_use_kg_per_ha", "N/A"),
                    "Soil Health Index": crop_data.get("soil_health_index", "N/A"),
                    "Adaptation Strategies": crop_data.get("adaptation_strategies", "N/A")
                })
            
            
            df = pd.DataFrame(results)
            st.write(df)
        else:
            st.warning(f"No top crops found for {selected_country}, but showing the top crops if available.")



    elif menu == "Top Economic Impact in country":
        st.subheader("Top Economic Impact Results")

        
        countries = list(db.countries_and_regions.find())
        country_names = [country["country"] for country in countries]
        selected_country = st.selectbox("Select Country", country_names)

        
        num_top_crops = st.slider("Select the number of top crops", 1, 10, 5)

        
        pipeline = [
            {
                "$lookup": {
                    "from": "crops",  # ربط جدول المحاصيل
                    "localField": "crop_id",  # الحقل في productivity_and_economic_impact الذي يربط مع المحاصيل
                    "foreignField": "_id",
                    "as": "crop_data"
                }
            },
            {
                "$unwind": "$crop_data"  # لفك الربط بين الـ crops
            },
            {
                "$lookup": {
                    "from": "climate",  # ربط جدول المناخ
                    "localField": "climate_id",  # الحقل في productivity_and_economic_impact الذي يربط مع المناخ
                    "foreignField": "_id",
                    "as": "climate_data"
                }
            },
            {
                "$unwind": "$climate_data"  # لفك الربط بين الـ climate
            },
            {
                "$lookup": {
                    "from": "countries_and_regions",  # ربط جدول البلدان
                    "localField": "country_region_id",  # الحقل في productivity_and_economic_impact الذي يربط مع countries_and_regions
                    "foreignField": "_id",
                    "as": "country_data"
                }
            },
            {
                "$unwind": "$country_data"  # لفك الربط بين الـ countries_and_regions
            },
            {
                "$match": {
                    "country_data.country": selected_country  # تصفية حسب الدولة المختارة
                }
            },
            {
                "$sort": {
                    "economic_impact_million_usd": -1  # ترتيب حسب economic_impact_million_usd
                }
            },
            {
                "$limit": num_top_crops  # أخذ أول N نتيجة حسب الـ slider
            }
        ]
        
        
        top_economic_impact_data = list(db.productivity_and_economic_impact.aggregate(pipeline))

        if top_economic_impact_data:
            
            results = []
            for data in top_economic_impact_data:
                crop_data = data.get("crop_data", {})
                climate_data = data.get("climate_data", {})
                country_data = data.get("country_data", {})
                
                results.append({
                    "Crop": crop_data.get("crop_type", "N/A"),
                    "Economic Impact (Million USD)": data.get("economic_impact_million_usd", "N/A"),
                    "Yield (MT/ha)": data.get("crop_yield_mt_per_ha", "N/A"),
                    "Year": climate_data.get("year", "N/A"),
                    "Temperature (°C)": climate_data.get("average_temperature_c", "N/A"),
                    "Country": country_data.get("country", "N/A"),
                    "Region": country_data.get("region", "N/A"),
                    "Precipitation (mm)": climate_data.get("total_precipitation_mm", "N/A"),
                    "CO2 Emissions (MT)": climate_data.get("co2_emissions_mt", "N/A"),
                    "Extreme Weather Events": climate_data.get("extreme_weather_events", "N/A"),
                    "Irrigation Access (%)": crop_data.get("irrigation_access_percent", "N/A"),
                    "Pesticide Use (kg/ha)": crop_data.get("pesticide_use_kg_per_ha", "N/A"),
                    "Fertilizer Use (kg/ha)": crop_data.get("fertilizer_use_kg_per_ha", "N/A"),
                    "Soil Health Index": crop_data.get("soil_health_index", "N/A"),
                    "Adaptation Strategies": crop_data.get("adaptation_strategies", "N/A")
                })
            
           
            df = pd.DataFrame(results)
            st.write(df)
        else:
            st.warning(f"No top economic impact crops found for {selected_country}.")


    elif menu == "Top crop yield in year":
        st.subheader("Top Crop Yield Results for a Specific Year")

        years = list(db.climate.distinct("year"))  # الحصول على قائمة من السنوات المتوفرة في جدول المناخ
        selected_year = st.selectbox("Select Year", sorted(years, reverse=True))  # عرض السنوات للمستخدم

       
        num_top_crops = st.slider("Select the number of top crops", 1, 20, 5)

        
        pipeline = [
            {
                "$lookup": {
                    "from": "crops",  # ربط جدول المحاصيل
                    "localField": "crop_id",  # الحقل في productivity_and_economic_impact الذي يربط مع المحاصيل
                    "foreignField": "_id",
                    "as": "crop_data"
                }
            },
            {
                "$unwind": "$crop_data"  # لفك الربط بين الـ crops
            },
            {
                "$lookup": {
                    "from": "climate",  # ربط جدول المناخ
                    "localField": "climate_id",  # الحقل في productivity_and_economic_impact الذي يربط مع المناخ
                    "foreignField": "_id",
                    "as": "climate_data"
                }
            },
            {
                "$unwind": "$climate_data"  # لفك الربط بين الـ climate
            },
            {
                "$match": {
                    "climate_data.year": selected_year  # تصفية حسب السنة المختارة
                }
            },
            {
                "$sort": {
                    "crop_yield_mt_per_ha": -1  # ترتيب حسب crop_yield_mt_per_ha
                }
            },
            {
                "$limit": num_top_crops  # أخذ أول N نتيجة حسب الـ slider
            }
        ]
        
        
        top_crop_yield_data = list(db.productivity_and_economic_impact.aggregate(pipeline))

        if top_crop_yield_data:
            
            results = []
            for data in top_crop_yield_data:
                crop_data = data.get("crop_data", {})
                climate_data = data.get("climate_data", {})
                
                results.append({
                    "Crop": crop_data.get("crop_type", "N/A"),
                    "Yield (MT/ha)": data.get("crop_yield_mt_per_ha", "N/A"),
                    "Economic Impact (Million USD)": data.get("economic_impact_million_usd", "N/A"),
                    "Year": climate_data.get("year", "N/A"),
                    "Temperature (°C)": climate_data.get("average_temperature_c", "N/A"),
                    "Precipitation (mm)": climate_data.get("total_precipitation_mm", "N/A"),
                    "CO2 Emissions (MT)": climate_data.get("co2_emissions_mt", "N/A"),
                    "Extreme Weather Events": climate_data.get("extreme_weather_events", "N/A"),
                    "Irrigation Access (%)": crop_data.get("irrigation_access_percent", "N/A"),
                    "Pesticide Use (kg/ha)": crop_data.get("pesticide_use_kg_per_ha", "N/A"),
                    "Fertilizer Use (kg/ha)": crop_data.get("fertilizer_use_kg_per_ha", "N/A"),
                    "Soil Health Index": crop_data.get("soil_health_index", "N/A"),
                    "Adaptation Strategies": crop_data.get("adaptation_strategies", "N/A")
                })
            
            
            df = pd.DataFrame(results)
            st.write(df)
        else:
            st.warning(f"No top crops found for the year {selected_year}.")


    elif menu == "Top Economic Impact in Year":
        st.subheader("Top Economic Impact Results for a Specific Year")

        
        years = list(db.climate.distinct("year"))  # الحصول على قائمة من السنوات المتوفرة في جدول المناخ
        selected_year = st.selectbox("Select Year", sorted(years, reverse=True))  # عرض السنوات للمستخدم

        
        num_top_impact = st.slider("Select the number of top impacts", 1, 10, 5)

       
        pipeline = [
            {
                "$lookup": {
                    "from": "crops",  # ربط جدول المحاصيل
                    "localField": "crop_id",  # الحقل في productivity_and_economic_impact الذي يربط مع المحاصيل
                    "foreignField": "_id",
                    "as": "crop_data"
                }
            },
            {
                "$unwind": "$crop_data"  # لفك الربط بين الـ crops
            },
            {
                "$lookup": {
                    "from": "climate",  # ربط جدول المناخ
                    "localField": "climate_id",  # الحقل في productivity_and_economic_impact الذي يربط مع المناخ
                    "foreignField": "_id",
                    "as": "climate_data"
                }
            },
            {
                "$unwind": "$climate_data"  # لفك الربط بين الـ climate
            },
            {
                "$match": {
                    "climate_data.year": selected_year  # تصفية حسب السنة المختارة
                }
            },
            {
                "$sort": {
                    "economic_impact_million_usd": -1  # ترتيب حسب economic_impact_million_usd
                }
            },
            {
                "$limit": num_top_impact  # أخذ أول N نتيجة حسب الـ slider
            }
        ]
        
        
        top_economic_impact_data = list(db.productivity_and_economic_impact.aggregate(pipeline))

        if top_economic_impact_data:
            results = []
            for data in top_economic_impact_data:
                crop_data = data.get("crop_data", {})
                climate_data = data.get("climate_data", {})
                
                results.append({
                    "Crop": crop_data.get("crop_type", "N/A"),
                    "Economic Impact (Million USD)": data.get("economic_impact_million_usd", "N/A"),
                    "Year": climate_data.get("year", "N/A"),
                    "Temperature (°C)": climate_data.get("average_temperature_c", "N/A"),
                    "Precipitation (mm)": climate_data.get("total_precipitation_mm", "N/A"),
                    "CO2 Emissions (MT)": climate_data.get("co2_emissions_mt", "N/A"),
                    "Extreme Weather Events": climate_data.get("extreme_weather_events", "N/A"),
                    "Irrigation Access (%)": crop_data.get("irrigation_access_percent", "N/A"),
                    "Pesticide Use (kg/ha)": crop_data.get("pesticide_use_kg_per_ha", "N/A"),
                    "Fertilizer Use (kg/ha)": crop_data.get("fertilizer_use_kg_per_ha", "N/A"),
                    "Soil Health Index": crop_data.get("soil_health_index", "N/A"),
                    "Adaptation Strategies": crop_data.get("adaptation_strategies", "N/A")
                })
            
            
            df = pd.DataFrame(results)
            st.write(df)
        else:
            st.warning(f"No top economic impacts found for the year {selected_year}.")






elif menu == "MapReduce":
    st.title("🧠 MapReduce Analytics")
    st.markdown("Select a MapReduce operation to analyze the climate & agriculture data.")

    # Button to prepare the data (add 'country' field)
    if st.button("🔄 Prepare Data (Add Country Field to Collections)"):
        with st.spinner("Updating collections with country names..."):
            def add_country_to_collection(collection_name):
                col = db[collection_name]
                updated = 0
                for doc in col.find():
                    if "country" not in doc:
                        country_region_id = doc.get("country_region_id")
                        if not country_region_id:
                            continue
                        country_doc = db.countries_and_regions.find_one({"_id": country_region_id})
                        if country_doc:
                            col.update_one({"_id": doc["_id"]}, {"$set": {"country": country_doc["country"]}})
                            updated += 1
                return updated

            updates = {
                "climate": add_country_to_collection("climate"),
                "crops": add_country_to_collection("crops"),
                "productivity_and_economic_impact": add_country_to_collection("productivity_and_economic_impact"),
            }

        st.success(f"✅ Updated documents: {updates}")

    # MapReduce analysis options
    options = {
        "🌡️ Avg Temperature per Country": ("climate", "average_temperature_c", "avg"),
        "🌧️ Total Precipitation per Country": ("climate", "total_precipitation_mm", "sum"),
        "🏭 Avg CO2 Emissions per Country": ("climate", "co2_emissions_mt", "avg"),
        "🌾 Avg Crop Yield per Country": ("productivity_and_economic_impact", "crop_yield_mt_per_ha", "avg"),
        "💰 Total Economic Impact per Country": ("productivity_and_economic_impact", "economic_impact_million_usd", "sum"),
    }

    choice = st.selectbox("🧪 Choose Analysis:", list(options.keys()))

    # Year and crop filters
    years = sorted(db.climate.distinct("year"))
    selected_year = st.selectbox("📅 Select Year (Optional):", ["All"] + years)

    crop_types = sorted(db.crops.distinct("crop_type"))
    selected_crop = st.selectbox("🌽 Select Crop Type (Optional):", ["All"] + crop_types)

    # MapReduce function
    def run_map_reduce(collection, field, operation="avg", out_name="result"):
        from bson.code import Code

        filters = {}
        if selected_year != "All":
            filters["year"] = selected_year
        if selected_crop != "All" and "crop" in collection:
            filters["crop_type"] = selected_crop

        collection_ref = db[collection]
        if filters:
            docs = list(collection_ref.find(filters))
            if not docs:
                return []  # Prevent error if no matching documents
            temp_name = f"temp_{collection}"
            db[temp_name].drop()
            db[temp_name].insert_many(docs)
            target_collection = temp_name
        else:
            target_collection = collection

        map_code = f"""
        function() {{
            if (this['{field}'] !== undefined && this.country) {{
                emit(this.country, {{ sum: this['{field}'], count: 1 }});
            }}
        }}
        """

        reduce_code = """
        function(key, values) {
            var result = { sum: 0, count: 0 };
            values.forEach(function(v) {
                result.sum += v.sum;
                result.count += v.count;
            });
            return result;
        }
        """

        if operation == "sum":
            finalize = Code("function(key, reducedVal) { return reducedVal.sum; }")
        else:
            finalize = Code("function(key, reducedVal) { return reducedVal.sum / reducedVal.count; }")

        db.command({
            "mapReduce": target_collection,
            "map": Code(map_code),
            "reduce": Code(reduce_code),
            "finalize": finalize,
            "out": out_name
        })

        return list(db[out_name].find())

    # Button to run MapReduce
    if st.button("🚀 Run MapReduce"):
        col, field, op = options[choice]
        with st.spinner("Running MapReduce..."):
            results = run_map_reduce(col, field, op, out_name="mapreduce_result")

        if results:
            df = pd.DataFrame(results)
            df.rename(columns={'_id': 'Country', 'value': choice}, inplace=True)
            st.dataframe(df)

            fig = px.bar(df, x="Country", y=choice, title=choice)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No matching data found. Try different filters.")










elif menu == "Update":
    st.subheader("✏️ Update Records by Multiple Filters")

    foreign_key_fields = ["country_region_id", "climate_id", "crop_id"]

    # ✅ دالة لجلب IDs من countries_and_regions حسب country أو region
    def get_country_region_ids_by_field(field, value):
        query = {field: value}
        matches = list(db_collections["Countries and Regions"].find(query))
        return [m["_id"] for m in matches]

    update_table = st.selectbox("Select table to update", list(db_collections.keys()))
    collection = db_collections[update_table]

    sample_docs = list(collection.find().limit(1))
    if not sample_docs:
        st.warning("No data available.")
    else:
        fields = [k for k in sample_docs[0].keys() if k not in ["_id"] + foreign_key_fields]

        # ✅ نضيف country و region فقط لو مش جدول countries_and_regions
        if update_table != "Countries and Regions":
            fields.extend(["country", "region"])

        selected_fields = st.multiselect("Select field(s) to filter by", fields)

        filter_query = {}
        country_filter_value = None  # نستخدمه لتحديد الفلترة الخاصة بـ region

        for field in selected_fields:
            # ----------- 1. تحديد القيم الممكنة -----------
            values = []

            if field == "country":
                values = db_collections["Countries and Regions"].distinct("country")

            elif field == "region":
                if country_filter_value:
                    region_docs = db_collections["Countries and Regions"].find({"country": country_filter_value})
                    values = sorted(set(doc["region"] for doc in region_docs))
                else:
                    values = db_collections["Countries and Regions"].distinct("region")

            else:
                values = collection.distinct(field)

            # ----------- 2. عرض القيم -----------
            display_map = {str(v): v for v in values}
            numeric_values = [v for v in values if isinstance(v, (int, float))]

            if numeric_values and field not in ["country", "region"]:
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                from_val, to_val = st.slider(
                    f"Range for {field}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1 if all(isinstance(v, int) for v in numeric_values) else 0.1,
                    key=f"range_filter_{field}"
                )
                filter_query[field] = {"$gte": from_val, "$lte": to_val}
            else:
                selected_value = st.selectbox(
                    f"Value for {field}",
                    sorted(display_map.keys()),
                    key=f"filter_{field}"
                )
                real_value = display_map[selected_value]

                if field == "country":
                    country_filter_value = real_value  # نحفظه علشان نفلتر بيه regions لاحقًا

                if field in ["country", "region"] and update_table != "Countries and Regions":
                    matching_ids = get_country_region_ids_by_field(field, real_value)
                    if matching_ids:
                        filter_query["country_region_id"] = {"$in": matching_ids}
                else:
                    filter_query[field] = real_value

        # ----------- 3. عرض السجلات المطابقة وتعديلها -----------
        if filter_query:
            matched_docs = list(collection.find(filter_query))
            if matched_docs:
                doc_options = [f"{str(doc['_id'])} - { {k: v for k, v in doc.items() if k != '_id'} }" for doc in matched_docs]
                selected_doc = st.selectbox("Select record to update", doc_options)
                selected_id = ObjectId(selected_doc.split(" - ")[0])
                record = collection.find_one({"_id": selected_id})

                if record:
                    st.markdown("### ✏️ Edit Fields")
                    updated_data = {}

                    for key, val in record.items():
                        if key == "_id" or key in foreign_key_fields:
                            continue  # لا نعدل الـ IDs الخارجية
                        if isinstance(val, (int, float)):
                            step_val = 1 if isinstance(val, int) else 0.01
                            new_val = st.number_input(f"{key}", value=val, step=step_val, key=f"edit_{key}")
                        else:
                            new_val = st.text_input(f"{key}", value=str(val), key=f"edit_{key}")
                        updated_data[key] = new_val

                    if st.button("Update Record"):
                        # تحويل القيم الرقمية تلقائيًا
                        for k, v in updated_data.items():
                            try:
                                updated_data[k] = eval(v) if isinstance(record[k], (int, float)) else v
                            except:
                                pass
                        collection.update_one({"_id": selected_id}, {"$set": updated_data})
                        st.success("✅ Record updated successfully!")
            else:
                st.info("No matching records found.")




# --------------------- Delete ----------------------
elif menu == "Delete":
    st.subheader("🗑️ Delete with Multiple Filters")

    foreign_key_fields = ["country_region_id", "climate_id", "crop_id"]

    # ✅ دالة لجلب country_region_id بناءً على country أو region
    def get_country_region_ids_by_field(field, value):
        # الاستعلام عن country_region_id بناءً على country أو region
        query = {field: value}
        matches = list(db_collections["Countries and Regions"].find(query))
        return [m["_id"] for m in matches]

    delete_table = st.selectbox("Select table to delete from", list(db_collections.keys()))
    collection = db_collections[delete_table]

    sample_docs = list(collection.find().limit(1))
    if not sample_docs:
        st.warning("No data available.")
    else:
        # ✅ إعداد الحقول
        fields = [k for k in sample_docs[0].keys() if k not in ["_id"] + foreign_key_fields]

        # ✅ إضافة country و region لكل الجداول ما عدا جدول Countries and Regions
        if delete_table != "Countries and Regions":
            fields.append("country")
            fields.append("region")

        selected_fields = st.multiselect("Select field(s) to filter by", fields)

        filter_query = {}
        country_filter_value = None  # لحفظ القيم المختارة لـ country للتصفية لاحقًا

        # ✅ إضافة الفلاتر الخاصة بـ country و region
        for field in selected_fields:
            values = []

            if field == "country":
                values = db_collections["Countries and Regions"].distinct("country")

            elif field == "region":
                if country_filter_value:
                    region_docs = db_collections["Countries and Regions"].find({"country": country_filter_value})
                    values = sorted(set(doc["region"] for doc in region_docs))
                else:
                    values = db_collections["Countries and Regions"].distinct("region")

            else:
                values = collection.distinct(field)

            # عرض القيم الممكنة للمستخدم
            display_map = {str(v): v for v in values}
            numeric_values = [v for v in values if isinstance(v, (int, float))]

            # التعامل مع القيم الرقمية
            if numeric_values and field not in ["country", "region"]:
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                selected_range = st.slider(
                    f"Select range for {field}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1 if all(isinstance(v, int) for v in numeric_values) else 0.1,
                    key=f"range_{field}"
                )
                filter_query[field] = {"$gte": selected_range[0], "$lte": selected_range[1]}
            else:
                selected_value = st.selectbox(
                    f"Value for {field}",
                    sorted(display_map.keys()),
                    key=f"filter_{field}"
                )
                real_value = display_map[selected_value]

                if field == "country":
                    country_filter_value = real_value  # حفظ البلد لاستخدامه في فلترة الـ region

                if field in ["country", "region"] and delete_table != "Countries and Regions":
                    matching_ids = get_country_region_ids_by_field(field, real_value)
                    if matching_ids:
                        filter_query["country_region_id"] = {"$in": matching_ids}
                else:
                    filter_query[field] = real_value

        # ✅ البحث عن السجلات المطابقة
        if filter_query:
            matched_docs = list(collection.find(filter_query))
            if matched_docs:
                doc_options = [
                    f"{str(doc['_id'])} - { {k: v for k, v in doc.items() if k != '_id'} }"
                    for doc in matched_docs
                ]
                selected_doc = st.selectbox("Select record to delete", doc_options)
                selected_id = ObjectId(selected_doc.split(" - ")[0])

                if selected_id:
                    record = collection.find_one({"_id": selected_id})
                    st.markdown("### 🧾 Record to be deleted:")
                    st.json(record)

                    confirm = st.checkbox("✅ I confirm I want to delete this record and related data")

                    if confirm and st.button("Confirm Delete"):
                        # ⚠️ حذف البيانات المرتبطة
                        if delete_table == "Countries and Regions":
                            db.productivity_and_economic_impact.delete_many({"country_region_id": selected_id})
                            db.crops.delete_many({"country_region_id": selected_id})
                            db.climate.delete_many({"country_region_id": selected_id})

                        elif delete_table == "Climate":
                            # إضافة حذف crops المرتبطة بـ climate_id
                            db.productivity_and_economic_impact.delete_many({"climate_id": selected_id})
                            db.crops.delete_many({"climate_id": selected_id})  # حذف المحاصيل المرتبطة بـ climate_id
                            db.climate.delete_one({"_id": selected_id})

                        elif delete_table == "Crops":
                            db.productivity_and_economic_impact.delete_many({"crop_id": selected_id})

                        elif delete_table == "Productivity and Economic Impact":
                            pass  # لا يوجد حذف بيانات مرتبطة مباشرة هنا

                        collection.delete_one({"_id": selected_id})
                        st.success("✅ Record and related data deleted successfully!")
            else:
                st.info("No matching records found.")
