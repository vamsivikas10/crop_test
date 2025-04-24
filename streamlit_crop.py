import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

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
        crop_query = """SELECT * FROM crop_data LIMIT 45015"""
        
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
st.title(":blue[Bird Species Analysis]")

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

# Sidebar Filters #
st.sidebar.header("Filters")

# Checkbox for year Filter and country Filter

apply_year_filter  = st.sidebar.checkbox("Apply Year filter", value=True)
selected_year = st.sidebar.multiselect("Select Year", crop["Year"].unique(), default=crop["Year"].unique())

apply_country_filter = st.sidebar.checkbox("Apply Country filter", value=True)
selected_country = st.sidebar.multiselect("Select Country", crop["Area"].unique(), default=crop["Area"].unique())

# Handle empty filters

selected_year = selected_year or crop["Year"].unique()
selected_country = selected_country or crop["Area"].unique()

# Apply filters dynamically based on checkboxes and selections

if apply_year_filter:
    filtered_crop = crop[crop["Year"].isin(selected_year)]
if apply_country_filter:
    filtered_crop = crop[crop["Area"].isin(selected_country)] 


#crop_final  = crop[crop["Year"].isin(selected_year) & crop["Country"].isin(selected_country)]

# Main analysis sections
if analysis_type == 'Analyze Crop Distribution':
    st.header("🌱 Crop Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Cultivated Crops")
        top_crops = filtered_crop.groupby('Item')['Area_Harvested'].sum().nlargest(10)
        fig, ax = plt.subplots()
        top_crops.plot(kind='barh', color=COLOR_1, ax=ax)
        ax.set_xlabel('Total Area Harvested (ha)')
        ax.set_ylabel('Crop')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Geographical Distribution")
        top_regions = filtered_crop.groupby('Area')['Production'].sum().nlargest(10)
        fig, ax = plt.subplots()
        top_regions.plot(kind='barh', color=COLOR_2, ax=ax)
        ax.set_xlabel('Total Production (tonnes)')
        ax.set_ylabel('Region')
        st.pyplot(fig)
        
    st.subheader("Crop Distribution by Region")
    pivot_data = filtered_crop.pivot_table(
        index='Area', 
        columns='Item', 
        values='Production', 
        aggfunc='mean'
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        np.log1p(pivot_data),  # Using log scale for better visualization
        cmap='YlOrBr',
        ax=ax
    )
    ax.set_title('Production Heatmap (Log Scale)')
    st.pyplot(fig)

elif analysis_type == 'Temporal Analysis':
    st.header("📈 Temporal Trends Analysis")
    
    # Let user select top N crops to display
    n_crops = st.slider("Select number of top crops to display", 5, 50, 10)
    
    # Calculate top crops by total production
    top_crops = filtered_crop.groupby('Item')['Production'].sum().nlargest(n_crops).index
    
    st.subheader(f"Yearly Production Trends (Top {n_crops} Crops)")
    yearly_trends = filtered_crop[filtered_crop['Item'].isin(top_crops)].groupby(['Year', 'Item'])['Production'].sum().unstack()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    yearly_trends.plot(ax=ax, linewidth=2)
    ax.set_ylabel('Production (tonnes)')
    ax.set_title(f'Production Trends of Top {n_crops} Crops', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # legend handling
    plt.legend(
        title='Crops',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize='small'  # Smaller font for more items
    )
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    st.pyplot(fig)

   # Yield Growth Analysis Section
    st.subheader("Yield Growth Analysis")
    
    # Step 1: Get top N crops by median yield (default N=10)
    top_n = st.slider(
        "Number of top crops to display", 
        min_value=5, 
        max_value=50, 
        value=10,
        key='yield_top_n'
    )
    
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

elif analysis_type == 'Environmental Relationships':
    st.header("🌍 Environmental Relationships")
    
    st.subheader("Area Harvested vs Yield")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=filtered_crop,
        x='Area_Harvested',
        y='Yield',
        hue='Item',
        alpha=0.6,
        ax=ax
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Resource Utilization vs Productivity')
    st.pyplot(fig)
    
    st.subheader("Yield Distribution by Crop")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=filtered_crop,
        x='Item',
        y='Yield',
        color=COLOR_1
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
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

# Add some space at the bottom
st.markdown("---")
st.markdown("### Data Summary")
st.write(f"Total records: {len(filtered_crop):,}")
st.write(f"Time period: {filtered_crop['Year'].min()} to {filtered_crop['Year'].max()}")
st.write(f"Regions: {len(filtered_crop['Area'].unique())}")
st.write(f"Crops: {len(filtered_crop['Item'].unique())}")
