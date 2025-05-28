import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Amref Kenya Future Funding Predictor - Kenya",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #f0f9ff 0%, #e0f7fa 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .donor-match {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data for demonstration
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # NGO characteristics
    sectors = ['Health', 'Education', 'Governance', 'Environment', 'Agriculture', 'Water & Sanitation']
    donor_types = ['USAID', 'DFID', 'UNICEF', 'World Bank', 'Gates Foundation', 'EU', 'SIDA', 'GIZ']
    counties = ['Nairobi', 'Mombasa', 'Turkana', 'Marsabit', 'Garissa', 'Kilifi', 'Nakuru', 'Kiambu']
    
    data = {
        'ngo_years_operation': np.random.randint(1, 25, n_samples),
        'past_projects_count': np.random.randint(0, 50, n_samples),
        'donor_retention_rate': np.random.uniform(0.2, 0.95, n_samples),
        'sector': np.random.choice(sectors, n_samples),
        'staff_count': np.random.randint(5, 200, n_samples),
        'compliance_score': np.random.uniform(0.5, 1.0, n_samples),
        'project_budget_requested': np.random.uniform(500000, 50000000, n_samples),
        'project_duration_months': np.random.randint(6, 60, n_samples),
        'beneficiaries_count': np.random.randint(100, 100000, n_samples),
        'target_county': np.random.choice(counties, n_samples),
        'donor_type': np.random.choice(donor_types, n_samples),
        'proposal_alignment_score': np.random.uniform(0.3, 1.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variables with realistic relationships
    # Funding amount (influenced by multiple factors)
    df['funding_received'] = (
        df['project_budget_requested'] * 0.3 +
        df['ngo_years_operation'] * 100000 +
        df['past_projects_count'] * 50000 +
        df['donor_retention_rate'] * 2000000 +
        df['compliance_score'] * 3000000 +
        df['proposal_alignment_score'] * 5000000 +
        np.random.normal(0, 500000, n_samples)
    ).clip(0, df['project_budget_requested'])
    
    # Approval probability
    approval_score = (
        df['ngo_years_operation'] * 0.02 +
        df['donor_retention_rate'] * 0.3 +
        df['compliance_score'] * 0.4 +
        df['proposal_alignment_score'] * 0.5 +
        np.random.normal(0, 0.1, n_samples)
    )
    df['approved'] = (approval_score > 0.6).astype(int)
    
    return df

# Load data
data = generate_sample_data()

# Sidebar for navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["📊 Dashboard", "🔮 Funding Predictor", "📈 Success Probability", "🎯 Donor Matching", "📋 Data Explorer", "📤 Upload & Analyze"]
)
# Main header
st.markdown('<h1 class="main-header">🎯 Amref Kenya Future Funding Prediction Model - Kenya</h1>', unsafe_allow_html=True)

if page == "📊 Dashboard":
    st.header("📊 Funding Landscape Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total NGOs Analyzed", 
            f"{len(data):,}",
            delta="Sample Dataset"
        )
    
    with col2:
        avg_funding = data['funding_received'].mean()
        st.metric(
            "Average Funding", 
            f"KES {avg_funding/1e6:.1f}M",
            delta=f"{data['approved'].mean()*100:.1f}% success rate"
        )
    
    with col3:
        top_sector = data['sector'].value_counts().index[0]
        st.metric(
            "Top Funded Sector", 
            top_sector,
            delta=f"{data['sector'].value_counts().iloc[0]} projects"
        )
    
    with col4:
        top_donor = data['donor_type'].value_counts().index[0]
        st.metric(
            "Most Active Donor", 
            top_donor,
            delta=f"{data['donor_type'].value_counts().iloc[0]} grants"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Funding by Sector")
        sector_funding = data.groupby('sector')['funding_received'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=sector_funding.values/1e6, 
            y=sector_funding.index,
            orientation='h',
            title="Average Funding by Sector (KES Millions)",
            color=sector_funding.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Success Rate by Donor Type")
        donor_success = data.groupby('donor_type')['approved'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=donor_success.values*100, 
            y=donor_success.index,
            orientation='h',
            title="Approval Rate by Donor (%)",
            color=donor_success.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Funding trends
    st.subheader("Funding Distribution Analysis")
    fig = px.histogram(
        data, 
        x='funding_received', 
        nbins=50,
        title="Distribution of Funding Amounts",
        color_discrete_sequence=['#2E8B57']
    )
    fig.update_layout(
        xaxis_title="Funding Amount (KES)",
        yaxis_title="Number of Projects"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Funding Predictor":
    st.header("🔮 Predict Your Funding Amount")
    st.write("Enter your NGO and project details to get a funding prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NGO Information")
        years_operation = st.slider("Years of Operation", 1, 25, 5)
        past_projects = st.slider("Number of Past Projects", 0, 50, 10)
        donor_retention = st.slider("Donor Retention Rate", 0.0, 1.0, 0.7, 0.1)
        sector = st.selectbox("Primary Sector", data['sector'].unique())
        staff_count = st.slider("Staff Count", 5, 200, 25)
        compliance_score = st.slider("Compliance Score", 0.5, 1.0, 0.8, 0.1)
    
    with col2:
        st.subheader("Project Details")
        budget_requested = st.number_input("Budget Requested (KES)", 500000, 50000000, 5000000, 500000)
        project_duration = st.slider("Project Duration (months)", 6, 60, 24)
        beneficiaries = st.number_input("Expected Beneficiaries", 100, 100000, 5000, 100)
        target_county = st.selectbox("Target County", data['target_county'].unique())
        donor_type = st.selectbox("Target Donor", data['donor_type'].unique())
        alignment_score = st.slider("Proposal-Donor Alignment", 0.3, 1.0, 0.8, 0.1)
    
    if st.button("🎯 Predict Funding", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'ngo_years_operation': [years_operation],
            'past_projects_count': [past_projects],
            'donor_retention_rate': [donor_retention],
            'sector': [sector],
            'staff_count': [staff_count],
            'compliance_score': [compliance_score],
            'project_budget_requested': [budget_requested],
            'project_duration_months': [project_duration],
            'beneficiaries_count': [beneficiaries],
            'target_county': [target_county],
            'donor_type': [donor_type],
            'proposal_alignment_score': [alignment_score]
        })
        
        # Encode categorical variables
        categorical_cols = ['sector', 'target_county', 'donor_type']
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(data[col])
            input_data[col] = le.transform(input_data[col])
        
        # Train model
        X = data.drop(['funding_received', 'approved'], axis=1)
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        y = data['funding_received']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        confidence_interval = prediction * 0.2  # Simulated confidence interval
        
        # Display results
        st.success("🎉 Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Predicted Funding",
                f"KES {prediction/1e6:.2f}M",
                delta=f"±{confidence_interval/1e6:.1f}M"
            )
        with col2:
            funding_ratio = (prediction / budget_requested) * 100
            st.metric(
                "Funding Ratio",
                f"{funding_ratio:.1f}%",
                delta="of requested amount")
        with col3:
            similar_projects = len(data[(data['sector'] == sector) & (data['donor_type'] == donor_type)])
            st.metric(
                "Similar Projects",
                similar_projects,
                delta="in our database")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Requested', 'Predicted', 'Best Case', 'Worst Case'],
            y=[budget_requested/1e6, prediction/1e6, (prediction + confidence_interval)/1e6, (prediction - confidence_interval)/1e6],
            marker_color=['#1f77b4', '#2E8B57', '#90EE90', '#FFA07A'],
            text=[f'{budget_requested/1e6:.1f}M', f'{prediction/1e6:.1f}M', 
                  f'{(prediction + confidence_interval)/1e6:.1f}M', f'{(prediction - confidence_interval)/1e6:.1f}M'],
            textposition='auto'
        ))
        fig.update_layout(
            title="Funding Prediction Analysis",
            xaxis_title="Scenario",
            yaxis_title="Amount (KES Millions)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Success Probability":
    st.header("📈 Proposal Success Probability")
    st.write("Analyze the likelihood of your proposal getting approved")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quick Assessment")
        org_maturity = st.select_slider("Organization Maturity", 
                                       options=["Startup", "Growing", "Established", "Mature"], 
                                       value="Growing")
        track_record = st.select_slider("Track Record", 
                                      options=["Poor", "Fair", "Good", "Excellent"], 
                                      value="Good")
        proposal_quality = st.select_slider("Proposal Quality", 
                                          options=["Basic", "Standard", "High", "Exceptional"], 
                                          value="High")
        donor_alignment = st.select_slider("Donor Alignment", 
                                         options=["Low", "Medium", "High", "Perfect"], 
                                         value="High")
    
    # Convert to numerical scores
    maturity_map = {"Startup": 0.2, "Growing": 0.5, "Established": 0.8, "Mature": 1.0}
    track_map = {"Poor": 0.2, "Fair": 0.5, "Good": 0.8, "Excellent": 1.0}
    quality_map = {"Basic": 0.3, "Standard": 0.6, "High": 0.8, "Exceptional": 1.0}
    alignment_map = {"Low": 0.3, "Medium": 0.6, "High": 0.8, "Perfect": 1.0}
    
    # Calculate probability
    base_probability = (
        maturity_map[org_maturity] * 0.3 +
        track_map[track_record] * 0.3 +
        quality_map[proposal_quality] * 0.25 +
        alignment_map[donor_alignment] * 0.15
    )
    
    # Add some randomness
    final_probability = min(base_probability + np.random.normal(0, 0.05), 1.0)
    final_probability = max(final_probability, 0.1)
    
    with col2:
        st.subheader("Success Probability")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = final_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Approval Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("📋 Recommendations")
        if final_probability < 0.4:
            st.error("🔴 Low success probability. Consider strengthening your proposal.")
            recommendations = [
                "Improve organizational capacity and track record",
                "Enhance proposal quality with detailed methodology",
                "Better align with donor priorities",
                "Consider partnering with established organizations"
            ]
        elif final_probability < 0.7:
            st.warning("🟡 Moderate success probability. Some improvements needed.")
            recommendations = [
                "Strengthen budget justification",
                "Add more concrete impact metrics",
                "Improve timeline feasibility",
                "Enhance monitoring & evaluation plan"
            ]
        else:
            st.success("🟢 High success probability. Strong proposal!")
            recommendations = [
                "Consider applying to multiple donors",
                "Prepare for due diligence process",
                "Plan for project implementation",
                "Document lessons learned for future proposals"
            ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

elif page == "🎯 Donor Matching":
    st.header("🎯 Smart Donor Matching")
    st.write("Find the best donor matches for your project")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Project Profile")
        project_sector = st.selectbox("Sector", data['sector'].unique(), key="donor_sector")
        project_budget = st.selectbox("Budget Range", 
                                    ["< 1M", "1M - 5M", "5M - 20M", "20M+"])
        target_region = st.selectbox("Target Region", data['target_county'].unique())
        project_type = st.selectbox("Project Type", 
                                  ["Service Delivery", "Capacity Building", "Research", "Advocacy"])
        
        if st.button("🔍 Find Matches", type="primary"):
            # Simulate donor matching algorithm
            donors_data = {
                'USAID': {'score': 85, 'avg_grant': '12M', 'focus': 'Health, Governance', 'success_rate': 72},
                'DFID': {'score': 78, 'avg_grant': '8M', 'focus': 'Education, Health', 'success_rate': 68},
                'Gates Foundation': {'score': 82, 'avg_grant': '15M', 'focus': 'Health, Agriculture', 'success_rate': 75},
                'World Bank': {'score': 71, 'avg_grant': '25M', 'focus': 'Infrastructure, Governance', 'success_rate': 65},
                'UNICEF': {'score': 88, 'avg_grant': '6M', 'focus': 'Education, Child Protection', 'success_rate': 80}
            }
            
            st.session_state.donor_matches = donors_data
    
    with col2:
        st.subheader("Recommended Donors")
        
        if 'donor_matches' in st.session_state:
            sorted_donors = sorted(st.session_state.donor_matches.items(), 
                                 key=lambda x: x[1]['score'], reverse=True)
            
            for i, (donor, info) in enumerate(sorted_donors[:3]):
                match_color = ["🥇", "🥈", "🥉"][i]
                
                with st.container():
                    st.markdown(f"""
                    <div class="donor-match">
                        <h4>{match_color} {donor} - {info['score']}% Match</h4>
                        <p><strong>Average Grant:</strong> KES {info['avg_grant']}</p>
                        <p><strong>Focus Areas:</strong> {info['focus']}</p>
                        <p><strong>Success Rate:</strong> {info['success_rate']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Matching visualization
            donor_names = [donor for donor, _ in sorted_donors]
            match_scores = [info['score'] for _, info in sorted_donors]
            
            fig = px.bar(
                x=donor_names,
                y=match_scores,
                title="Donor Match Scores",
                color=match_scores,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Set your project details and click 'Find Matches' to see recommendations")

elif page == "📋 Data Explorer":
    st.header("📋 Data Explorer")
    st.write("Explore the underlying data and model insights")
    
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "🔍 Feature Analysis", "📈 Model Performance"])
    
    with tab1:
        st.subheader("Dataset Sample")
        st.dataframe(data.head(100), height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Statistics")
            st.write(data.describe())
        
        with col2:
            st.subheader("Data Quality")
            quality_metrics = {
                'Total Records': len(data),
                'Missing Values': data.isnull().sum().sum(),
                'Duplicate Records': data.duplicated().sum(),
                'Data Completeness': f"{((len(data) - data.isnull().sum().sum()) / (len(data) * len(data.columns)) * 100):.1f}%"
            }
            for metric, value in quality_metrics.items():
                st.metric(metric, value)
    
    with tab2:
        st.subheader("Feature Importance Analysis")
        
        # Calculate feature importance
        X = data.drop(['funding_received', 'approved'], axis=1)
        categorical_cols = ['sector', 'target_county', 'donor_type']
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        y = data['funding_received']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="Feature Importance for Funding Prediction"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlations")
        correlation_matrix = X.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Model Performance Metrics")
        
        # Train models and show performance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Regression model
        reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Absolute Error", f"KES {mae/1e6:.2f}M")
            st.metric("Root Mean Square Error", f"KES {rmse/1e6:.2f}M")
        
        # Classification model for approval prediction
        y_class = data['approved']
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
        
        class_model = RandomForestClassifier(n_estimators=100, random_state=42)
        class_model.fit(X_train_c, y_train_c)
        y_pred_c = class_model.predict(X_test_c)
        
        accuracy = accuracy_score(y_test_c, y_pred_c)
        
        with col2:
            st.metric("Classification Accuracy", f"{accuracy*100:.1f}%")
        
        # Prediction vs Actual scatter plot
        fig = px.scatter(
            x=y_test/1e6,
            y=y_pred/1e6,
            title="Predicted vs Actual Funding (Millions KES)",
            labels={'x': 'Actual Funding', 'y': 'Predicted Funding'}
        )
        fig.add_trace(go.Scatter(
            x=[y_test.min()/1e6, y_test.max()/1e6],
            y=[y_test.min()/1e6, y_test.max()/1e6],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        st.plotly_chart(fig, use_container_width=True)
elif page == "📤 Upload & Analyze":
    st.header("📤 Upload & Analyze Excel Data")
    st.write("Upload your NGO/project data in Excel format for comprehensive analysis")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Read the Excel file
            df_uploaded = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.uploaded_data = df_uploaded
            
            # Data validation
            st.subheader("🔍 Data Validation Report")
            
            # Check for required columns
            required_columns = [
                'ngo_years_operation', 'past_projects_count', 'donor_retention_rate',
                'sector', 'project_budget_requested', 'donor_type'
            ]
            
            missing_columns = [col for col in required_columns if col not in df_uploaded.columns]
            
            if missing_columns:
                st.error(f"⚠️ Missing required columns: {', '.join(missing_columns)}")
            else:
                st.success("✅ All required columns present")
                
                # Data quality metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records Uploaded", len(df_uploaded))
                with col2:
                    missing_values = df_uploaded.isnull().sum().sum()
                    st.metric("Missing Values", missing_values)
                with col3:
                    duplicates = df_uploaded.duplicated().sum()
                    st.metric("Duplicate Records", duplicates)
                
                # Sample data display
                st.subheader("📋 Data Sample")
                st.dataframe(df_uploaded.head())
                
                # Analysis section
                st.subheader("📊 Automated Analysis")
                
                # Prepare data for modeling
                try:
                    # Encode categorical variables
                    categorical_cols = ['sector', 'donor_type']
                    for col in categorical_cols:
                        if col in df_uploaded.columns:
                            le = LabelEncoder()
                            df_uploaded[col] = le.fit_transform(df_uploaded[col])
                    
                    # Train models on original data
                    X = data.drop(['funding_received', 'approved'], axis=1)
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                    
                    y_funding = data['funding_received']
                    y_approved = data['approved']
                    
                    # Train models
                    funding_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    funding_model.fit(X, y_funding)
                    
                    approval_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    approval_model.fit(X, y_approved)
                    
                    # Make predictions
                    try:
                        # Select only columns that exist in both datasets
                        common_cols = [col for col in X.columns if col in df_uploaded.columns]
                        
                        # Predict funding
                        funding_predictions = funding_model.predict(df_uploaded[common_cols])
                        df_uploaded['predicted_funding'] = funding_predictions
                        
                        # Predict approval probability
                        approval_probs = approval_model.predict_proba(df_uploaded[common_cols])[:, 1]
                        df_uploaded['approval_probability'] = approval_probs
                        
                        # Display results
                        st.success("✅ Analysis complete!")
                        
                        # Summary metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            avg_funding = df_uploaded['predicted_funding'].mean()
                            st.metric(
                                "Average Predicted Funding", 
                                f"KES {avg_funding/1e6:.2f}M"
                            )
                        with col2:
                            avg_approval = df_uploaded['approval_probability'].mean() * 100
                            st.metric(
                                "Average Approval Probability", 
                                f"{avg_approval:.1f}%"
                            )
                        
                        # Visualizations
                        tab1, tab2, tab3 = st.tabs(["📈 Funding Predictions", "✅ Approval Probabilities", "📋 Full Analysis"])
                        
                        with tab1:
                            st.subheader("Funding Prediction Distribution")
                            fig = px.histogram(
                                df_uploaded,
                                x='predicted_funding',
                                nbins=20,
                                title="Distribution of Predicted Funding Amounts",
                                labels={'predicted_funding': 'Predicted Funding (KES)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Top/bottom projects
                            st.subheader("Projects with Highest/Lowest Predicted Funding")
                            top_projects = df_uploaded.nlargest(5, 'predicted_funding')
                            bottom_projects = df_uploaded.nsmallest(5, 'predicted_funding')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("🏆 Top Projects")
                                st.dataframe(top_projects[['sector', 'donor_type', 'predicted_funding']]
                                            .sort_values('predicted_funding', ascending=False)
                                            .assign(predicted_funding=lambda x: x['predicted_funding'].apply(lambda y: f"KES {y/1e6:.2f}M")))
                            
                            with col2:
                                st.write("⚠️ Projects Needing Improvement")
                                st.dataframe(bottom_projects[['sector', 'donor_type', 'predicted_funding']]
                                            .sort_values('predicted_funding')
                                            .assign(predicted_funding=lambda x: x['predicted_funding'].apply(lambda y: f"KES {y/1e6:.2f}M")))
                        
                        with tab2:
                            st.subheader("Approval Probability Analysis")
                            
                            # Gauge for average probability
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=avg_approval,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Average Approval Probability"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'steps': [
                                        {'range': [0, 40], 'color': "lightgray"},
                                        {'range': [40, 70], 'color': "yellow"},
                                        {'range': [70, 100], 'color': "green"}],
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # By sector
                            if 'sector' in df_uploaded.columns:
                                st.subheader("Approval Probability by Sector")
                                sector_approval = df_uploaded.groupby('sector')['approval_probability'].mean().sort_values()
                                fig = px.bar(
                                    x=sector_approval.values * 100,
                                    y=sector_approval.index,
                                    orientation='h',
                                    title="Average Approval Probability by Sector (%)",
                                    labels={'x': 'Approval Probability', 'y': 'Sector'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with tab3:
                            st.subheader("Full Analysis Report")
                            
                            # Feature importance
                            st.write("### Key Factors Influencing Predictions")
                            feature_importance = pd.DataFrame({
                                'feature': X.columns,
                                'importance': funding_model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            fig = px.bar(
                                feature_importance.head(10),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title="Top 10 Factors Affecting Funding Amount"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Recommendations
                            st.write("### Actionable Recommendations")
                            
                            # Funding recommendations
                            st.write("#### To Increase Funding Potential:")
                            funding_recs = [
                                "Focus on improving donor retention rates",
                                "Increase compliance scores through better documentation",
                                "Align proposals more closely with donor priorities",
                                "Build track record with smaller projects first",
                                "Target donors with higher average funding amounts"
                            ]
                            for rec in funding_recs:
                                st.write(f"- {rec}")
                            
                            # Approval recommendations
                            st.write("#### To Improve Approval Chances:")
                            approval_recs = [
                                "Ensure complete and accurate proposal submissions",
                                "Demonstrate clear organizational capacity",
                                "Provide detailed monitoring and evaluation plans",
                                "Show strong beneficiary impact metrics",
                                "Highlight past project successes"
                            ]
                            for rec in approval_recs:
                                st.write(f"- {rec}")
                            
                            # Download button for full report
                            st.download_button(
                                label="📥 Download Full Analysis Report",
                                data=df_uploaded.to_csv(index=False).encode('utf-8'),
                                file_name='ngo_funding_analysis_report.csv',
                                mime='text/csv'
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
                        st.write("Please ensure your data columns match the expected format")
                
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 Amref Kenya Future Funding Prediction Model for Kenya | Built with Streamlit & Machine Learning</p>
    <p><em>This is a demonstration model. Actual predictions should be based on comprehensive real-world data.</em></p>
</div>
""", unsafe_allow_html=True)