import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats 

# Database configuration
config = {
    'host': 'mysql-3cfdc572-avamsi2k11-4e7b.h.aivencloud.com',
    'user': 'avnadmin',
    'password': 'AVNS_ozTCcwHYoNvj53twyRY',
    'port': 22799
}

# Global variables
crop_data_df = None

def fetch_data(conn, sql_query):
    """Fetch data from MySQL using a given connection"""
    try:
        cursor = conn.cursor()
        cursor.execute("USE project_guvi_Crop_analysis")
        cursor.execute(sql_query)
        
        rows = cursor.fetchall()
        columns = [i[0] for i in cursor.description]
        
        cursor.close()
        
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    except Error as e:
        print(f"Error fetching data: {e}")
        return None

def create_connection():
    """Create a database connection"""
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def initialize():
    global crop_data_df  # Declare as global
    
    # Create connection
    conn = create_connection()
    if conn is None:
        return
    
    try:
        # Queries
        crop_query = """SELECT 
                            c.data_id,
                            a.Area,
                            i.Item,
                            y.Year,
                            c.Area_Harvested,
                            c.Yield,
                            c.Production
                        FROM 
                            CROP_data c
                        JOIN 
                            areas a ON c.Area_id = a.Area_id
                        JOIN 
                            items i ON c.Item_id = i.Item_id
                        JOIN 
                            years y ON c.year_id = y.year_id
                        LIMIT 44000"""
        
        crop_data_df = fetch_data(conn, crop_query)
    
    finally:
        if conn.is_connected():
            conn.close()
            print("MySQL connection is closed")

if __name__ == "__main__":
    initialize()


# Set color scheme
COLOR_1='#C8920B'
COLOR_2='#FFB233'
TEXT_COLOR = '#2C1D4C'  
BACKGROUND_COLOR = '#F0F2F6'
WHITE = '#FFFFFF'

# Set matplotlib style
plt.style.use('ggplot')

crop=crop_data_df

# Main title 
st.title(":blue[Crop Analysis]")

# Sidebar options
analysis_type = st.sidebar.radio(
    'Analysis Type:', 
    [
        'Analyze Crop Distribution',
        'Temporal Analysis',
        'Environmental Relationships',
        'Input-Output Relationships',
        'Comparative Analysis',
        'Outliers and Anomalies'
    ]
)


sidebar_checkbox = st.sidebar.checkbox("Prediction")

if sidebar_checkbox:
        df = crop_data_df 
        # ======================
        # 1. DATA PREPROCESSING
        # ======================
        # Handle outliers
        numeric_cols = ['Area_Harvested', 'Yield', 'Production']
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z_scores < 3).all(axis=1)]

        # Feature engineering
        df['Area_Yield_Interaction'] = df['Area_Harvested'] * df['Yield']

        # Define features and target
        X = df[['Area_Harvested', 'Yield', 'Year', 'Item', 'Area', 'Area_Yield_Interaction']]
        y = df['Production']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ======================
        # 2. PREPROCESSING SETUP
        # ======================
        numeric_features = ['Area_Harvested', 'Yield', 'Year', 'Area_Yield_Interaction']
        categorical_features = ['Item', 'Area']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # ======================
        # 3. MODEL PIPELINES
        # ======================
        # Linear Regression Pipeline
        lr_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Random Forest Pipeline
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Decision Tree Pipeline
        dt_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42))
        ])

        # Polynomial Regression Pipeline
        poly_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', LinearRegression())
        ])

        # Train all models
        pipelines = {
            "Linear Regression": lr_pipeline,
            "Random Forest": rf_pipeline,
            "Decision Tree": dt_pipeline,
            "Polynomial Regression": poly_pipeline
        }

        for name, pipeline in pipelines.items():
            pipeline.fit(X_train, y_train)

        # ======================
        # 4. STREAMLIT APP
        # ======================
        st.title("🌾 Crop Production Predictor")

        # Model Evaluation
        st.subheader("📈 Model Evaluation on Test Data")
        results = []
        for name, pipeline in pipelines.items():
            y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results.append({
                "Model": name,
                "R² Score": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE":rmse
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        best_model_name = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
        st.success(f"✅ Best model based on test data R² Score: {best_model_name}")

        # ======================
        # 5. PREDICTION INTERFACE
        # ======================
        st.subheader("🔮 Production Prediction")

        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
            area_harvested = st.number_input("Area_Harvested (hectares)", min_value=1, value=100)
            yield_val = st.number_input("Yield (tons/hectare)", min_value=0.0, value=2.5, step=0.1)

        with col2:
            item = st.selectbox("Crop", options=sorted(df['Item'].unique()))
            area = st.selectbox("Area", options=sorted(df['Area'].unique()))
            #price_per_ton = st.number_input("Price per Ton (USD)", value=250.0)

        if st.button("Predict Production"):
            # Create input DataFrame with all required features
            input_data = pd.DataFrame({
                'Area_Harvested': [area_harvested],
                'Yield': [yield_val],
                'Year': [year],
                'Item': [item],
                'Area': [area],
                'Area_Yield_Interaction': [area_harvested * yield_val]
            })
            
            st.markdown("### 📊 Prediction Results")
            
            # Get predictions from all models
            for name, pipeline in pipelines.items():
                prediction = pipeline.predict(input_data)[0]
                prediction = max(0, prediction)  # Ensure non-negative
                value = prediction * price_per_ton
                
                st.write(f"**{name}:**")
                st.write(f"- Production: {prediction:.2f} tons")
                st.write(f"- Estimated Value: ${value:,.2f} USD")
                st.write("---")

        # ======================
        # 6. ADDITIONAL FEATURES
        # ======================
        st.subheader("🌱 Farming Recommendations")

        if item and area:
            st.write(f"### Precision farming tips for {item} in {area}:")
            st.write("- Use soil sensors to optimize irrigation")
            st.write("- Implement drone-based crop monitoring")
            st.write("- Consider AI-powered pest detection systems")
        else:
            st.info("Select a crop and area to get farming recommendations")
    

# Sidebar Filters
st.sidebar.header("Filters")

# Checkboxes and selectors
apply_year_filter = st.sidebar.checkbox("Apply Year filter", value=True)
selected_year = st.sidebar.multiselect("Select Year", options=crop["Year"].unique(), default=crop["Year"].unique())

apply_country_filter = st.sidebar.checkbox("Apply Country filter", value=True)
selected_country = st.sidebar.multiselect("Select Country", options=crop["Area"].unique(), default=crop["Area"].unique())

# Initialize filtered data
filtered_crop = crop.copy()

# Apply filters if enabled (handle empty selections too)
if apply_year_filter:
    selected_year = selected_year or crop["Year"].unique()  # Default to all years
    filtered_crop = filtered_crop[filtered_crop["Year"].isin(selected_year)]

if apply_country_filter:
    selected_country = selected_country or crop["Area"].unique()  # Default to all countries
    filtered_crop = filtered_crop[filtered_crop["Area"].isin(selected_country)]

# # Display filtered results
# st.write(f"Filtered to {len(filtered_crop)} records")



# Main analysis sections
if analysis_type == 'Analyze Crop Distribution':
        
        st.header("🌱 Crop Distribution Analysis")
    
        ##(1)##
        st.subheader("Top Cultivated Crops")
        top_crops = filtered_crop.groupby('Item')['Area_Harvested'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_crops.plot(kind='barh', color=COLOR_1, ax=ax)
        ax.set_xlabel('Total Area Harvested (ha)')
        ax.set_ylabel('Crop')
        st.pyplot(fig)
    
        ##(2)##
        st.subheader("Geographical Distribution")
        top_regions = filtered_crop.groupby('Area')['Production'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_regions.plot(kind='barh', color=COLOR_2, ax=ax)
        ax.set_xlabel('Total Production (tonnes)')
        ax.set_ylabel('Region')
        st.pyplot(fig)

        ##(3)##
        st.subheader("Crop Distribution by Region (Interactive Heatmap)")
        
        # Create pivot table: rows = regions, columns = crops, values = mean production
        pivot_data2 = filtered_crop.pivot_table( index='Area',columns='Item',values='Production',aggfunc='mean').fillna(0)
        
        # Apply log1p transform for better color scaling
        log_pivot = np.log1p(pivot_data2)
        
        # Create interactive heatmap with Plotly Express
        fig = px.imshow(
            log_pivot,
            labels=dict(x="Crop", y="Region", color="Log(Mean Production + 1)"),
            x=log_pivot.columns,
            y=log_pivot.index,
            color_continuous_scale='YlOrBr',
            aspect='auto',
            origin='lower'  # So that regions start from bottom if you prefer
        )
        
        fig.update_layout(title="Crop Production Heatmap by Region (Log Scale)", height=900 ,  width=900 ,
                           margin = {'l': 80, 'r': 80, 't': 100, 'b': 80 } )
        
        # Show Plotly figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)


 ##################################################################################################################################################################################################################
elif analysis_type == 'Temporal Analysis':
   
    st.header("📈 Temporal Trends Analysis")
    
    # Let user select top N crops to display
    n_crops = st.slider("Select number of top crops to display", 5, 50, 10)
    
    ##(1)##
    
    # Calculate top crops by total production
    top_p_crops = filtered_crop.groupby('Item')['Production'].sum().nlargest(n_crops).index
    
    st.subheader(f"Yearly Production Trends (Top {n_crops} Crops)")
    yearly_trends = filtered_crop[filtered_crop['Item'].isin(top_p_crops)].groupby(['Year', 'Item'])['Production'].sum().unstack()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    yearly_trends.plot(ax=ax, linewidth=2)
    ax.set_ylabel('Production (tonnes)')
    ax.set_title(f'Production Trends of Top {n_crops} Crops', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # legend handling
    plt.legend(title='Crops', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,fontsize='small' )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)

    ##(2)##
    # Calculate top crops by total area harvested
    top_ah_crops = filtered_crop.groupby('Item')['Area_Harvested'].sum().nlargest(n_crops).index
    
    st.subheader(f"Yearly Area_Harvested Trends (Top {n_crops} Crops)")
    yearly_trends = filtered_crop[filtered_crop['Item'].isin(top_ah_crops)].groupby(['Year', 'Item'])['Area_Harvested'].sum().unstack()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    yearly_trends.plot(ax=ax, linewidth=2)
    ax.set_ylabel('Area_Harvested(tonnes)')
    ax.set_title(f'Area_Harvested Trends of Top {n_crops} Crops', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # legend handling
    plt.legend(title='Crops', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,fontsize='small' )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)

    ##(3)##
    # Calculate top crops by total yield 
    top_crops = filtered_crop.groupby('Item')['Yield'].sum().nlargest(n_crops).index
    
    st.subheader(f"Yearly Yield Trends (Top {n_crops} Crops)")
    yearly_trends = filtered_crop[filtered_crop['Item'].isin(top_crops)].groupby(['Year', 'Item'])['Yield'].sum().unstack()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    yearly_trends.plot(ax=ax, linewidth=2)
    ax.set_ylabel('Yield(tonnes)')
    ax.set_title(f'Yield Trends of Top {n_crops} Crops', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # legend handling
    plt.legend(title='Crops', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,fontsize='small' )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)

    ##(4)##
    
    # Yield Growth Analysis Section
    st.subheader("Yield Growth Analysis")
    
    # Step 1: Get top N crops by median yield (default N=10)
    top_n = st.slider( "Number of top crops to display",  min_value=5, max_value=50, value=10, key='yield_top_n')
    
    # Calculate top crops (using median to avoid outlier skew)
    top_crops = filtered_crop.groupby('Item')['Yield'].median().nlargest(top_n)
    
    # Step 2: Create selectbox without default selection
    if not top_crops.empty:
        selected_crop_trend = st.selectbox(
            "Select Crop for Analysis",
            options=top_crops.index.tolist(),
            key='yield_crop_select'
        )
        
        # Step 3: Filter data and plot
        crop_trend = filtered_crop[filtered_crop['Item'] == selected_crop_trend]
        
              # Pivot to Year vs Area matrix
        heatmap_data = crop_trend.pivot_table(
            index='Year', 
            columns='Area', 
            values='Yield',
            aggfunc='median'
        )
        
        # Plotly interactive heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Region", y="Year", color="Yield"),
            color_continuous_scale='YlOrBr',
            aspect='auto'
        )
        fig.update_layout(
            title=f'Yield Trends for {selected_crop_trend} (200 Regions)',
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)
                
        # Show summary stats
        with st.expander("📊 View summary statistics"):
            st.dataframe(
                crop_trend.groupby('Area')['Yield'].describe()
                .style.background_gradient(cmap='YlOrBr')
            )
    else:
        st.warning("No crops available for analysis.")

    ############(5)
    
    st.subheader("Growth Analysis: Yield/Production Trends")
    
    # User selects metric and grouping
    metric = st.selectbox("Metric to analyze", ["Yield", "Production"], key="growth_metric")
    group_by = st.selectbox("Analyze trend by", ["Item", "Area"], key="growth_group")
    
    if group_by.startswith("Item"):
     group_col = "Item"
    else:
     group_col = "Area"
    
    trend_results = []
    
    for group, group_df in filtered_crop.groupby(group_col):
        group_df = group_df.sort_values("Year")
        if group_df["Year"].nunique() >= 2:
            first_year = group_df["Year"].min()
            last_year = group_df["Year"].max()
            first_value = group_df.loc[group_df["Year"] == first_year, metric].mean()
            last_value = group_df.loc[group_df["Year"] == last_year, metric].mean()
            if first_value != 0:
                pct_change = 100 * (last_value - first_value) / abs(first_value)
            else:
                pct_change = np.nan  # Avoid division by zero
            trend_results.append({
                group_col: group,
                "first_year": first_year,
                "last_year": last_year,
                f"{metric}_start": first_value,
                f"{metric}_end": last_value,
                "pct_change": pct_change,
                "years": group_df["Year"].nunique()
            })
    
    trend_df = pd.DataFrame(trend_results).dropna(subset=["pct_change"])
    
    if not trend_df.empty:
        st.write(f"Top 10 {group_by} with **increasing** {metric}:")
        st.dataframe(trend_df.sort_values("pct_change", ascending=False).head(10)[[group_col, "first_year", "last_year", f"{metric}_start", f"{metric}_end", "pct_change"]])
    
        st.write(f"Top 10 {group_by} with **decreasing** {metric}:")
        st.dataframe(trend_df.sort_values("pct_change", ascending=True).head(10)[[group_col, "first_year", "last_year", f"{metric}_start", f"{metric}_end", "pct_change"]])
    
        # Optional: Plot trend for a selected crop/region
        selected = st.selectbox(f"Select {group_by} to visualize trend", trend_df[group_col])
        plot_df = filtered_crop[filtered_crop[group_col] == selected].sort_values("Year")
        fig = px.line(plot_df, x="Year", y=metric, markers=True, title=f"{metric} Trend for {selected}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to compute trends for the selected grouping.")

        
##################################################################################################################################################################################################################

elif analysis_type == 'Environmental Relationships':
    st.header("🌍 Environmental Relationships")
    
    st.subheader("Yield vs Area")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Hexbin for density (handles overlapping points)
    hexbin = ax.hexbin(
        x=filtered_crop['Area_Harvested'],
        y=filtered_crop['Yield'],
        gridsize=40,
        cmap='YlOrBr',
        bins='log',
        mincnt=1,
        alpha=0.7
    )
          # Overlay a sample of points for context
    sample_df = filtered_crop.sample(1000)  # Avoid overplotting
    sns.scatterplot(
        data=sample_df,
        x='Area_Harvested',
        y='Yield',
        color='red',
        alpha=0.3,
        s=20,
        ax=ax,
        label='Sample Points'
    )

    
    ax.set_xscale('log')
    ax.set_yscale('log')
    cb = fig.colorbar(hexbin)
    cb.set_label('Log10(Data Density)')
    plt.legend()
    st.pyplot(fig)
    
elif analysis_type == 'Input-Output Relationships':
    st.header("⚙️ Input-Output Relationships")
    
    st.subheader("Correlation Matrix")
    numeric_cols = filtered_crop.select_dtypes(include=np.number).columns
    corr_matrix = filtered_crop[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='YlOrBr',
        center=0,
        ax=ax
    )
    st.pyplot(fig)
    
    st.subheader("Production vs Area Harvested")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        data=filtered_crop,
        x='Area_Harvested',
        y='Production',
        scatter_kws={'alpha':0.3},
        line_kws={'color':'red'},
        ax=ax
    )
    ax.set_title('Production vs Cultivated Area')
    st.pyplot(fig)
##################################################################################################################################################################################################################
elif analysis_type == 'Comparative Analysis':
    st.header("📊 Comparative Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Yielding Crops")
        top_yield = filtered_crop.groupby('Item')['Yield'].median().nlargest(10)
        fig, ax = plt.subplots()
        top_yield.plot(kind='barh', color=COLOR_1, ax=ax)
        ax.set_xlabel('Median Yield (hg/ha)')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top Producing Regions")
        top_producers = filtered_crop.groupby('Area')['Production'].sum().nlargest(10)
        fig, ax = plt.subplots()
        top_producers.plot(kind='barh', color=COLOR_2, ax=ax)
        ax.set_xlabel('Total Production (tonnes)')
        st.pyplot(fig)
        
    st.subheader("Productivity Analysis")
    filtered_crop['Productivity'] = filtered_crop['Production'] / filtered_crop['Area_Harvested']
    productive_regions = filtered_crop.groupby(['Area', 'Item'])['Productivity'].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    productive_regions.nlargest(10, productive_regions.columns[0]).plot(
        kind='bar',
        ax=ax,
        color=[COLOR_1, COLOR_2]
    )
    ax.set_ylabel('Productivity (Production/Area)')
    st.pyplot(fig)
##################################################################################################################################################################################################################
elif analysis_type == 'Outliers and Anomalies':
    st.header("🔍 Outliers and Anomalies Detection")
    
    st.subheader("Yield Outliers by Crop")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=filtered_crop,
        x='Item',
        y='Yield',
        color=COLOR_1
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)
    
    st.subheader("Anomaly Detection")
    # Using IQR method to detect outliers
    Q1 = filtered_crop['Yield'].quantile(0.25)
    Q3 = filtered_crop['Yield'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = 1.5 * IQR
    outliers = filtered_crop[
        (filtered_crop['Yield'] < (Q1 - outlier_threshold)) | 
        (filtered_crop['Yield'] > (Q3 + outlier_threshold))
    ]
    
    st.write(f"Found {len(outliers)} yield outliers (IQR method):")
    st.dataframe(outliers.sort_values('Yield', ascending=False))
    
    # Plot outliers
    if not outliers.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=filtered_crop,
            x='Year',
            y='Yield',
            color=COLOR_1,
            alpha=0.5,
            ax=ax
        )
        sns.scatterplot(
            data=outliers,
            x='Year',
            y='Yield',
            color='red',
            ax=ax
        )
        ax.set_title('Yield Outliers Highlighted')
        st.pyplot(fig)

#####################################################
st.markdown("---")
st.markdown("### Data Summary")
st.write(f"Total records: {len(filtered_crop):,}")
st.write(f"Time period: {filtered_crop['Year'].min()} to {filtered_crop['Year'].max()}")
st.write(f"Regions: {len(filtered_crop['Area'].unique())}")
st.write(f"Crops: {len(filtered_crop['Item'].unique())}")


       
