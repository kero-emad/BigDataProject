import streamlit as st
from pymongo import MongoClient
import pandas as pd
import numpy as np
from bson import ObjectId
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Mongo connection
client = MongoClient("mongodb://localhost:27017/")
db = client["agriculture_climate_db"]

# Collections
db_collections = {
    "Countries and Regions": db.countries_and_regions,
    "Climate": db.climate,
    "Crops": db.crops,
    "Productivity and Economic Impact": db.productivity_and_economic_impact
}

st.set_page_config(page_title="Agriculture DB Admin", layout="wide")
st.title("🌾 Climate Change Impact on Agriculture")

# Sidebar menu
menu = st.sidebar.selectbox("Select Action", ["Introduction","Parsing","Insert","Update", "Select","Search","Delete"])

# Helper: Get Country-Region options
def get_country_region_options():
    data = db.countries_and_regions.find()
    return [f"{doc['country']} - {doc['region']}" for doc in data]

def get_country_region_id(country_region_str):
    country, region = country_region_str.split(" - ")
    result = db.countries_and_regions.find_one({"country": country, "region": region})
    return result["_id"] if result else None
# Helper: Get Year options
def get_year_options():
    data = db.climate.find()
    return sorted(set(doc['year'] for doc in data), reverse=True)

# Helper: Get Crop Type options
def get_crop_type_options():
    data = db.crops.find()
    return sorted(set(doc['crop_type'] for doc in data))

# Helper: Get Productivity and Economic Impact by Year
def get_productivity_by_year():
    data = db.productivity_and_economic_impact.find()
    return sorted(set(doc['year'] for doc in data), reverse=True)


if menu == "Introduction":
    st.title("🌍 Climate & Agriculture Data Explorer")

    # صورة توضيحية (قم بتعديل المسار حسب الصورة التي تريد إدراجها)
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
    st.title("📦 Database Tables & Relationships")

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

            # Insert climate
            climate_doc = {
                'year': int(row['Year']),
                'average_temperature_c': row['Average_Temperature_C'],
                'total_precipitation_mm': row['Total_Precipitation_mm'],
                'co2_emissions_mt': row['CO2_Emissions_MT'],
                'extreme_weather_events': row['Extreme_Weather_Events'],
                'country_region_id': country_region_id
            }
            climate_id = climate_collection.insert_one(climate_doc).inserted_id

            # Insert crop
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

            # Insert productivity
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
        year = st.number_input("Year", step=1)
        avg_temp = st.number_input("Average Temperature (°C)")
        precipitation = st.number_input("Total Precipitation (mm)")
        co2 = st.number_input("CO2 Emissions (MT)")
        extreme = st.text_input("Extreme Weather Events")
        
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
        crop_type = st.text_input("Crop Type")
        irrigation = st.number_input("Irrigation Access (%)")
        pesticide = st.number_input("Pesticide Use (KG/HA)")
        fertilizer = st.number_input("Fertilizer Use (KG/HA)")
        soil = st.text_input("Soil Health Index")
        strategy = st.text_input("Adaptation Strategies")
        if st.button("Insert"):
            cr_id = get_country_region_id(cr_str)
            db.crops.insert_one({
                "crop_type": crop_type,
                "irrigation_access_percent": irrigation,
                "pesticide_use_kg_per_ha": pesticide,
                "fertilizer_use_kg_per_ha": fertilizer,
                "soil_health_index": soil,
                "adaptation_strategies": strategy,
                "country_region_id": cr_id
            })
            st.success("Inserted successfully!")

    elif insert_table == "Productivity and Economic Impact":
        st.subheader("Add Productivity & Economic Data")
        cr_str = st.selectbox("Country & Region", get_country_region_options())
        yield_ = st.number_input("Crop Yield (MT/HA)")
        econ = st.number_input("Economic Impact (Million USD)")
        
        # تعديل هذا الجزء:
        if cr_str:
          cr_id = get_country_region_id(cr_str)
          climate_docs = list(db.climate.find({"country_region_id": cr_id}))
          climate_ids = {}
          for c in climate_docs:
              year = c.get("year")
              if year not in climate_ids:
                    climate_ids[str(year)] = c["_id"]
        
          if climate_ids:
            climate_str = st.selectbox("Climate Reference (Year)", list(climate_ids.keys()))
          else:
            st.warning("No climate data available for the selected country/region.")
            climate_str = None
        if st.button("Insert") and climate_str:
            climate_id = climate_ids[climate_str]
            db.productivity_and_economic_impact.insert_one({
            "crop_yield_mt_per_ha": yield_,
            "economic_impact_million_usd": econ,
            "country_region_id": cr_id,
            "climate_id": climate_id
        })
            st.success("Inserted successfully!")
# ------------------ SELECT -------------------
elif menu == "Select":
    table_name = st.selectbox("Choose table to view", list(db_collections.keys()))
    collection = db_collections[table_name]
    
    # Load data
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    if df.empty:
        st.warning("No data found in this table.")
    else:
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        # Slider to choose number of rows to view
        num_rows = st.slider(
            "Select number of rows to display", 
            min_value=1, 
            max_value=len(df), 
            value=min(10, len(df))
        )
        
        # Display selected rows
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
        "View Countries Producing a Specific Crop",
        "Fertilizer Usage Analysis",
        "Pesticide Usage Analysis"
    ])

    # ========== ✅ Search by Country & Year ==========
    if search_method == "Search by Country & Year":
        country_region_str = st.selectbox("Select Country & Region", get_country_region_options())

        if country_region_str:
            cr_id = get_country_region_id(country_region_str)

            climate_years = sorted({doc.get("year") for doc in db.climate.find({"country_region_id": cr_id}, {"year": 1}) if "year" in doc})

            if climate_years:
                selected_year = st.selectbox("Select Year", climate_years)

                # 🌡️ Climate Data
                st.markdown("### 🌡️ Climate Data")
                climate_data = list(db.climate.find({"country_region_id": cr_id, "year": selected_year}))
                if climate_data:
                    st.dataframe(pd.DataFrame(climate_data).drop(columns=["_id", "country_region_id"]), use_container_width=True)
                else:
                    st.info("No climate data found.")

                # جمع كل climate_id للسنة المحددة
                climate_ids = [doc["_id"] for doc in climate_data]

                # 🌾 Crop Data
                st.markdown("### 🌾 Crop Data")
                crop_data = list(db.crops.find({"climate_id": {"$in": climate_ids}, "country_region_id": cr_id}))
                if crop_data:
                    st.dataframe(pd.DataFrame(crop_data).drop(columns=["_id", "country_region_id", "climate_id"]), use_container_width=True)
                else:
                    st.info("No crop data found for selected year and region.")

                # 📈 Productivity & Economic Impact
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

            # كل أنواع المحاصيل لهذا البلد
            crops_cursor = db.crops.find({"country_region_id": cr_id})
            crop_options = sorted({doc["crop_type"] for doc in crops_cursor if "crop_type" in doc})

            if crop_options:
                selected_crop = st.selectbox("Select Crop Type", crop_options)

                # استرجاع وثائق المحصول المطابقة
                selected_crops = list(db.crops.find({"country_region_id": cr_id, "crop_type": selected_crop}))

                if selected_crops:
                    crop_ids = [crop["_id"] for crop in selected_crops]

                    # 🌡️ Climate Data المرتبطة بمحاصيل الإنتاج
                    st.markdown("### 🌡️ Climate Data")
                    climate_ids = db.productivity_and_economic_impact.distinct("climate_id", {"crop_id": {"$in": crop_ids}})
                    climates = list(db.climate.find({"_id": {"$in": climate_ids}}))
                    if climates:
                        st.dataframe(pd.DataFrame(climates).drop(columns=["_id", "country_region_id"]), use_container_width=True)
                    else:
                        st.info("No climate data found.")

                    # 🌾 Crop Data
                    st.markdown("### 🌾 Crop Data")
                    st.dataframe(pd.DataFrame(selected_crops).drop(columns=["_id", "country_region_id"]), use_container_width=True)

                    # 📈 Productivity & Economic Impact
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


          # ========== ✅ Search by Year & Crop Type ==========
    elif search_method == "Search by Year & Crop Type":
        years = sorted({doc.get("year") for doc in db.climate.find({}, {"year": 1}) if "year" in doc})
        crops_available = sorted({doc.get("crop_type") for doc in db.crops.find({}, {"crop_type": 1}) if "crop_type" in doc})

        selected_year = st.selectbox("Select Year", years)
        selected_crop_type = st.selectbox("Select Crop Type", crops_available)

        # 🔍 Get crops with selected type
        matching_crops = list(db.crops.find({"crop_type": selected_crop_type}))
        crop_ids = [c["_id"] for c in matching_crops]

        # 🔍 Get productivity entries for selected year and crop type
        matching_productivity = list(db.productivity_and_economic_impact.find({
            "crop_id": {"$in": crop_ids}
        }))

        # Filter productivity by climate year
        filtered_productivity = []
        for prod in matching_productivity:
            climate_doc = db.climate.find_one({"_id": prod["climate_id"]})
            if climate_doc and climate_doc.get("year") == selected_year:
                prod["climate"] = climate_doc
                filtered_productivity.append(prod)

        if filtered_productivity:
            # 🌡️ Climate Table with country_region name
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

            # 🌾 Crop Data
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

            # 📈 Productivity
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

    # اختيار الدولة والمنطقة
        country_region_names = {
            str(doc["_id"]): f"{doc['country']} - {doc['region']}"
            for doc in db_collections["Countries and Regions"].find()
        }

        selected_cr_id = st.selectbox("Choose Country & Region", list(country_region_names.keys()), format_func=lambda x: country_region_names[x])

    # اختيار العامل المناخي
        climate_factors = {
            "Average Temperature (°C)": "average_temperature_c",
            "Total Precipitation (mm)": "total_precipitation_mm",
            "CO2 Emissions (Mt)": "co2_emissions_mt",
            "Extreme Weather Events": "extreme_weather_events"
        }

        selected_factor_label = st.selectbox("Choose Climate Factor", list(climate_factors.keys()))
        selected_factor = climate_factors[selected_factor_label]

    # جلب بيانات المناخ
        climate_docs = list(db_collections["Climate"].find({"country_region_id": ObjectId(selected_cr_id)}))
        climate_df = pd.DataFrame(climate_docs)

        if not climate_df.empty:
            climate_df = climate_df[["_id", "year", selected_factor]]
            climate_df = climate_df.dropna(subset=["year", selected_factor])
            climate_df.rename(columns={"_id": "climate_id"}, inplace=True)

        # جلب بيانات الإنتاجية الاقتصادية المرتبطة
            impact_docs = list(db_collections["Productivity and Economic Impact"].find({"country_region_id": ObjectId(selected_cr_id)}))
            impact_df = pd.DataFrame(impact_docs)

            if not impact_df.empty:
                impact_df = impact_df[["climate_id", "crop_yield_mt_per_ha"]]
                impact_df = impact_df.dropna(subset=["crop_yield_mt_per_ha"])

            # دمج البيانات
                merged_df = pd.merge(climate_df, impact_df, on="climate_id")

                if not merged_df.empty:
                    # تجميع حسب السنة
                    grouped = merged_df.groupby("year").agg({
                        selected_factor: "mean",
                        "crop_yield_mt_per_ha": "mean"
                    }).reset_index()

                    st.write("### Yearly Averages")
                    st.dataframe(grouped)

                # رسم المخطط
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

                # عرض معامل الارتباط
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

    # اختيار الدولة والمنطقة
        country_region_names = {
            str(doc["_id"]): f"{doc['country']} - {doc['region']}"
            for doc in db_collections["Countries and Regions"].find()
        }

        selected_cr_id = st.selectbox("Choose Country & Region", list(country_region_names.keys()), format_func=lambda x: country_region_names[x])

    # اختيار العامل الزراعي
        crop_factors = {
            "Irrigation Access (%)": "irrigation_access_percent",
            "Pesticide Use (kg/ha)": "pesticide_use_kg_per_ha",
            "Fertilizer Use (kg/ha)": "fertilizer_use_kg_per_ha",
            "Soil Health Index": "soil_health_index"
        }

        selected_crop_label = st.selectbox("Choose Crop Factor", list(crop_factors.keys()))
        selected_crop_factor = crop_factors[selected_crop_label]

    # جلب بيانات المحاصيل
        crops_docs = list(db_collections["Crops"].find({"country_region_id": ObjectId(selected_cr_id)}))
        crops_df = pd.DataFrame(crops_docs)

        if not crops_df.empty:
            crops_df = crops_df[["_id", selected_crop_factor]]
            crops_df.rename(columns={"_id": "crop_id"}, inplace=True)
            crops_df = crops_df.dropna(subset=[selected_crop_factor])

        # جلب بيانات الإنتاجية الاقتصادية المرتبطة
            impact_docs = list(db_collections["Productivity and Economic Impact"].find({"country_region_id": ObjectId(selected_cr_id)}))
            impact_df = pd.DataFrame(impact_docs)

            if not impact_df.empty:
                impact_df = impact_df[["crop_id", "crop_yield_mt_per_ha"]]
                impact_df = impact_df.dropna(subset=["crop_yield_mt_per_ha"])

            # دمج البيانات
                merged_df = pd.merge(crops_df, impact_df, on="crop_id")

                if not merged_df.empty:
                    st.write("### Data Summary")
                    st.dataframe(merged_df)

                # رسم المخطط
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

                # حساب معامل الارتباط
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
        # الحصول على المحاصيل التي من النوع المحدد
            matched_crops = list(db.crops.find({"crop_type": selected_crop}))

        # استخراج country_region_id من المحاصيل
            region_ids = list({crop["country_region_id"] for crop in matched_crops if "country_region_id" in crop})

        # جلب بيانات الدول/المناطق المرتبطة
            countries = list(db.countries_and_regions.find({"_id": {"$in": region_ids}}))

        # عرض النتائج في جدول
            if countries:
                df = pd.DataFrame(countries)
                df = df.rename(columns={"country": "Country", "region": "Region"})
                st.dataframe(df[["Country", "Region"]])
            else:
                st.info("No countries found for the selected crop.")

    elif search_method == "Fertilizer Usage Analysis":
        st.subheader("🔍 Find Years and Regions with High Fertilizer Use")

    # شريط تمرير لاختيار الكمية
        threshold = st.slider("Minimum fertilizer use (kg/ha)", 0, 100, 50)

    # استعلام للحصول على السجلات التي تجاوزت الحد
        matching_docs = list(db.crops.find({"fertilizer_use_kg_per_ha": {"$gt": threshold}}))

        results = []

        for crop in matching_docs:
            cr_id = crop.get("country_region_id")
            climate_id = crop.get("climate_id")

        # جلب بيانات الدولة/المنطقة
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

    # شريط تمرير لاختيار الكمية
        pesticide_threshold = st.slider("Minimum pesticide use (kg/ha)", 0, 50, 10)

    # استعلام للحصول على السجلات التي تجاوزت الحد
        matching_docs = list(db.crops.find({"pesticide_use_kg_per_ha": {"$gt": pesticide_threshold}}))

        results = []

        for crop in matching_docs:
            cr_id = crop.get("country_region_id")
            climate_id = crop.get("climate_id")

        # جلب بيانات الدولة/المنطقة
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




