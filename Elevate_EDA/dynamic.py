import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from io import StringIO
import plotly.figure_factory as ff
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import sys
warnings.filterwarnings('ignore')

# Increase the recursion limit
sys.setrecursionlimit(10000)

@st.cache_data
def load_data(file):
    try:
        # Use pandas to read the file in chunks
        chunks = pd.read_csv(file, chunksize=10000)
        df = pd.concat(chunks, ignore_index=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def get_unique_values(df, column, max_values=100):
    try:
        unique_values = df[column].unique()
        if len(unique_values) > max_values:
            return np.random.choice(unique_values, max_values, replace=False)
        return unique_values
    except Exception as e:
        st.warning(f"Error getting unique values for {column}: {str(e)}")
        return []

def main():
    # Set page configuration with a PNG logo as the page icon
    st.set_page_config(
    page_title="Elevate EDA",
    page_icon="Elevate_EDA.png",  # Specify the path to your PNG logo
    layout="wide")

    # Load and display the logo in the sidebar
    logo_path = "logo.png"  # Specify the path to your logo
    
    st.sidebar.image(logo_path, use_column_width=True)
    
    st.title("Robust EDA Tool for Large Datasets")
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(":file_folder: Upload a CSV file", type="csv")

    if uploaded_file is not None:
        file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
        st.write(f"File size: {file_size:.2f} MB")

        df = load_data(uploaded_file)
        if df is None:
            st.stop()
        
        st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("Columns:", df.columns.tolist())

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Sidebar filters
        st.sidebar.header("Choose your filters:")
        filters = {}
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns for performance
            unique_values = get_unique_values(df, col)
            filters[col] = st.sidebar.multiselect(f"Select {col}", unique_values)

        # Apply filters
        for col, values in filters.items():
            if values:
                df = df[df[col].isin(values)]

        # Data overview
        st.subheader("Data Overview")
        if st.checkbox("Show data types"):
            st.write(df.dtypes)
        if st.checkbox("Show summary statistics"):
            st.write(df.describe())
        if st.checkbox("Show missing values"):
            st.write(df.isnull().sum())

        # Column selection for analysis
        st.subheader("Select columns for analysis")
        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", numeric_cols)

        # Efficient data sampling for visualization
        sample_size = min(10000, df.shape[0])
        df_sample = df.sample(n=sample_size)

        # Visualizations
        st.subheader("Data Visualization")
        chart_type = st.selectbox("Select chart type", ["Bar", "Scatter", "Line", "Histogram", "Box", "Violin", "Density Contour"])

        try:
            if chart_type == "Bar":
                fig = px.bar(df_sample, x=x_axis, y=y_axis, title=f"Bar Chart: {x_axis} vs {y_axis}")
            elif chart_type == "Scatter":
                fig = px.scatter(df_sample, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}")
            elif chart_type == "Line":
                fig = px.line(df_sample, x=x_axis, y=y_axis, title=f"Line Chart: {x_axis} vs {y_axis}")
            elif chart_type == "Histogram":
                fig = px.histogram(df_sample, x=y_axis, title=f"Histogram: {x_axis}")
            elif chart_type == "Box":
                fig = px.box(df_sample, x=x_axis, y=y_axis, title=f"Box Plot: {x_axis} vs {y_axis}")
            elif chart_type == "Violin":
                fig = px.violin(df_sample, x=x_axis, y=y_axis, title=f"Violin Plot: {x_axis} vs {y_axis}")
            elif chart_type == "Density Contour":
                fig = px.density_contour(df_sample, x=x_axis, y=y_axis, title=f"Density Contour: {x_axis} vs {y_axis}")

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")

        # Correlation analysis
        if len(numeric_cols) > 1:
            st.subheader("Correlation Analysis")
            corr_method = st.selectbox("Select correlation method", ["Pearson", "Spearman"])
            try:
                corr = df[numeric_cols].corr(method=corr_method.lower())
                fig = px.imshow(corr, text_auto=True, aspect="auto", title=f"{corr_method} Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in correlation analysis: {str(e)}")

        # Time series analysis (if date columns are present)
        if date_cols:
            st.subheader("Time Series Analysis")
            date_col = st.selectbox("Select date column", date_cols)
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                time_series_df = df.groupby(df[date_col].dt.to_period('D'))[y_axis].mean().reset_index()
                time_series_df[date_col] = time_series_df[date_col].dt.to_timestamp()
                fig = px.line(time_series_df, x=date_col, y=y_axis, title=f"Time Series: {date_col} vs {y_axis}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in time series analysis: {str(e)}")

        # Enhanced K-Means Clustering
        if st.checkbox("Perform K-Means Clustering"):
            st.subheader("K-Means Clustering")
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
            cluster_features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:2])
            
            if cluster_features:
                try:
                    cluster_data = df_sample[cluster_features]
                    
                    # Standardize the features
                    scaler = StandardScaler()
                    cluster_data_scaled = scaler.fit_transform(cluster_data)
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    df_sample['Cluster'] = kmeans.fit_predict(cluster_data_scaled)
                    
                    # Create scatter plot
                    fig = px.scatter(df_sample, x=cluster_features[0], y=cluster_features[1], color='Cluster', title="K-Means Clustering")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summarize clusters
                    st.subheader("Cluster Summary")
                    st.write("This section provides an overview of the key characteristics for each cluster. For each feature, you can see the average value and the variation (standard deviation) within the cluster.")

                    for i in range(n_clusters):
                        cluster_data = df_sample[df_sample['Cluster'] == i]
                        st.write(f"**Cluster {i} Summary:**")
                        
                        # List of feature summaries for the current cluster
                        summary = []
                        for feature in cluster_features:
                            avg_value = cluster_data[feature].mean()
                            std_value = cluster_data[feature].std()
                            summary.append(f"**{feature}**: Average = {avg_value:.2f}, Variation = ±{std_value:.2f}")
                        
                        # Display summary as a list
                        st.write("\n".join(summary))

                    # Interpret clusters
                    st.subheader("Cluster Interpretation")
                    st.write("This section provides an interpretation of each cluster, describing the overall trends in feature values relative to the dataset average.")

                    # Get the number of samples in each cluster and the cluster centers (unscaled)
                    cluster_sizes = df_sample['Cluster'].value_counts().sort_index()
                    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

                    # Interpret each cluster
                    for i in range(n_clusters):
                        st.write(f"**Cluster {i} Interpretation:**")
                        
                        # Basic cluster size info
                        interpretation = f"- **Cluster Size**: {cluster_sizes[i]} samples ({cluster_sizes[i]/len(df_sample)*100:.1f}% of the dataset)"
                        
                        # Add interpretations for each feature based on whether the cluster center is above or below the mean
                        characteristics = []
                        for j, feature in enumerate(cluster_features):
                            if cluster_centers[i][j] > np.mean(df_sample[feature]):
                                characteristics.append(f"higher-than-average **{feature}**")
                            else:
                                characteristics.append(f"lower-than-average **{feature}**")
                        
                        # Combine the interpretation and display it
                        interpretation += "\n- **Key Characteristics**: " + ", ".join(characteristics)
                        st.write(interpretation)

                    
                except Exception as e:
                    st.error(f"Error in clustering: {str(e)}")

        # Data Profiling
        if st.checkbox("Generate Data Profile"):
            st.subheader("Data Profiling")
            try:
                profile_df = pd.DataFrame({
                    'Column': df.columns,
                    'Non-Null Count': df.count(),
                    'Dtype': df.dtypes,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.write(profile_df)
            except Exception as e:
                st.error(f"Error in data profiling: {str(e)}")

        # Display raw data (with pagination)
        st.subheader("Raw Data Sample")
        page_size = st.slider("Rows per page", min_value=10, max_value=100, value=50)
        page_number = st.number_input("Page number", min_value=1, value=1)
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        st.write(df_sample.iloc[start_idx:end_idx])

        # Download filtered data
        if st.button("Download Filtered Data Sample"):
            csv = df_sample.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Click here to download",
                data=csv,
                file_name="filtered_data_sample.csv",
                mime="text/csv",
            )

    else:
        st.write("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()
