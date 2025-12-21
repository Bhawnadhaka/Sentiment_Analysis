"""
Streamlit Web Dashboard
Central UI for the entire MLOps pipeline
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
from datetime import datetime
import time
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="MLOps Sentiment Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
DB_PATH = "sentiment_predictions.db"

# Helper functions
@st.cache_data(ttl=60)
def get_api_health():
    """Check API health"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.json()
    except:
        return {"status": "unhealthy", "message": "API not reachable"}

@st.cache_data(ttl=60)
def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=2)
        return response.json()
    except:
        return {}

@st.cache_data(ttl=30)
def get_predictions_from_db():
    """Load predictions from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT 
                title,
                sentiment,
                confidence,
                source,
                topic,
                processed_at
            FROM predictions
            ORDER BY processed_at DESC
            LIMIT 1000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['processed_at'] = pd.to_datetime(df['processed_at'])
        return df
    except:
        return pd.DataFrame()

def predict_sentiment(text):
    """Make prediction via API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=5
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/machine-learning.png", width=100)
    st.title("🚀 MLOps Dashboard")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔮 Live Predictions", "📊 Analytics", "🎯 Model Performance", "⚙️ System Status", "📡 Real-Time Stream"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # API Status
    health = get_api_health()
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    st.metric("API Status", f"{status_color} {health.get('status', 'unknown')}")
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit")

# Main content based on page selection
if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">🚀 MLOps Sentiment Analysis Pipeline</h1>', unsafe_allow_html=True)
    
    # Key metrics
    df = get_predictions_from_db()
    model_info = get_model_info()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Predictions", len(df))
    
    with col2:
        if not df.empty:
            positive_pct = (df['sentiment'] == 'positive').sum() / len(df) * 100
            st.metric("😊 Positive %", f"{positive_pct:.1f}%")
    
    with col3:
        if not df.empty:
            avg_confidence = df['confidence'].mean()
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        cache_stats = model_info.get('cache', {})
        if cache_stats.get('enabled'):
            hit_rate = cache_stats.get('hit_rate', 0)
            st.metric("⚡ Cache Hit Rate", f"{hit_rate:.1f}%")
    
    st.divider()
    
    # Recent predictions
    st.subheader("📝 Recent Predictions")
    if not df.empty:
        recent = df.head(10)[['title', 'sentiment', 'confidence', 'topic', 'processed_at']]
        recent['confidence'] = recent['confidence'].apply(lambda x: f"{x:.2%}")
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Start the streaming pipeline!")
    
    # Sentiment trend
    if not df.empty and len(df) > 10:
        st.subheader("📈 Sentiment Trend Over Time")
        df_hourly = df.copy()
        df_hourly['hour'] = df_hourly['processed_at'].dt.floor('H')
        trend = df_hourly.groupby(['hour', 'sentiment']).size().reset_index(name='count')
        
        fig = px.line(trend, x='hour', y='count', color='sentiment',
                     title='Sentiment Distribution Over Time',
                     color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B'})
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Live Predictions":
    st.header("🔮 Live Sentiment Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text to Analyze")
        
        # Quick examples
        example = st.selectbox(
            "Quick Examples:",
            [
                "Custom text...",
                "This product is amazing! Best purchase ever!",
                "Terrible service. Very disappointed.",
                "The AI revolution is transforming industries.",
                "Stock market crash causes panic among investors."
            ]
        )
        
        if example != "Custom text...":
            text_input = example
        else:
            text_input = ""
        
        text = st.text_area(
            "Your text:",
            value=text_input,
            height=150,
            placeholder="Enter any text to analyze sentiment..."
        )
        
        predict_btn = st.button("🎯 Analyze Sentiment", type="primary", use_container_width=True)
        
        if predict_btn and text:
            with st.spinner("Analyzing..."):
                result = predict_sentiment(text)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    probs = result['probabilities']
                    
                    # Display result
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        emoji = "😊" if sentiment == "positive" else "😟"
                        st.markdown(f"### {emoji} Sentiment: **{sentiment.upper()}**")
                        st.progress(confidence)
                        st.caption(f"Confidence: {confidence:.2%}")
                    
                    with col_b:
                        st.markdown("### Probability Distribution")
                        prob_df = pd.DataFrame([probs]).T.reset_index()
                        prob_df.columns = ['Sentiment', 'Probability']
                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(prob_df, hide_index=True)
    
    with col2:
        st.subheader("📊 Model Info")
        info = get_model_info()
        
        if info:
            st.json({
                "Model Type": info.get('model_type', 'N/A'),
                "Device": info.get('device', 'N/A'),
                "Classes": info.get('classes', []),
                "Cache Enabled": info.get('cache', {}).get('enabled', False)
            })
        
        st.subheader("⚡ Performance Tips")
        st.info("""
        - Cached queries are 10x faster
        - Longer texts take more time
        - Best results with clear sentiment
        """)

elif page == "📊 Analytics":
    st.header("📊 Analytics Dashboard")
    
    df = get_predictions_from_db()
    
    if df.empty:
        st.warning("No data available. Run the streaming pipeline to generate predictions.")
    else:
        # Overall statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Articles", len(df))
            st.metric("Unique Sources", df['source'].nunique())
        
        with col2:
            positive = (df['sentiment'] == 'positive').sum()
            negative = (df['sentiment'] == 'negative').sum()
            st.metric("Positive", positive, delta=None)
            st.metric("Negative", negative, delta=None)
        
        with col3:
            avg_conf = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
            st.metric("Data Points", len(df))
        
        st.divider()
        
        # Visualizations
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            fig1 = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B'}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_b:
            st.subheader("Top Topics")
            topic_counts = df['topic'].value_counts().head(10)
            fig2 = px.bar(
                x=topic_counts.values,
                y=topic_counts.index,
                orientation='h',
                title='Articles by Topic'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        
        # Confidence distribution
        st.subheader("Confidence Score Distribution")
        fig3 = px.histogram(
            df,
            x='confidence',
            color='sentiment',
            nbins=20,
            color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B'}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Top sources
        st.subheader("Top News Sources")
        source_sentiment = df.groupby(['source', 'sentiment']).size().reset_index(name='count')
        top_sources = df['source'].value_counts().head(10).index
        source_sentiment = source_sentiment[source_sentiment['source'].isin(top_sources)]
        
        fig4 = px.bar(
            source_sentiment,
            x='source',
            y='count',
            color='sentiment',
            barmode='group',
            color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B'}
        )
        st.plotly_chart(fig4, use_container_width=True)

elif page == "🎯 Model Performance":
    st.header("🎯 Model Performance Metrics")
    
    model_info = get_model_info()
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Configuration")
            st.json({
                "Type": model_info.get('model_type', 'N/A'),
                "Device": model_info.get('device', 'N/A'),
                "Classes": model_info.get('classes', [])
            })
        
        with col2:
            st.subheader("Cache Statistics")
            cache = model_info.get('cache', {})
            if cache.get('enabled'):
                st.metric("Status", "✅ Enabled")
                st.metric("Hit Rate", f"{cache.get('hit_rate', 0):.1f}%")
                st.metric("Total Connections", cache.get('total_connections', 0))
            else:
                st.warning("Cache is disabled")
    
    st.divider()
    
    # Training metrics (from MLflow)
    st.subheader("📈 Training Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "76.45%")
    with col2:
        st.metric("F1 Score", "76.68%")
    with col3:
        st.metric("Precision", "76.78%")
    with col4:
        st.metric("Recall", "76.45%")
    
    # Evidently reports
    st.divider()
    st.subheader("📊 Monitoring Reports")
    
    reports_dir = Path("monitoring/evidently_reports")
    if reports_dir.exists():
        reports = list(reports_dir.glob("*.html"))
        if reports:
            for report in sorted(reports, reverse=True)[:4]:
                with st.expander(f"📄 {report.stem}"):
                    st.write(f"Generated: {datetime.fromtimestamp(report.stat().st_mtime)}")
                    st.write(f"Size: {report.stat().st_size / 1024:.1f} KB")
                    if st.button(f"Open {report.name}", key=report.name):
                        st.info(f"Open this file in your browser: {report}")
        else:
            st.info("No reports generated yet. Run: `python run_pipeline.py dashboard`")

elif page == "⚙️ System Status":
    st.header("⚙️ System Status & Health")
    
    # Service status
    col1, col2, col3 = st.columns(3)
    
    services = {
        "FastAPI": f"{API_URL}/health",
        "Prometheus": "http://localhost:9090/-/healthy",
        "Grafana": "http://localhost:3000/api/health"
    }
    
    for i, (service, url) in enumerate(services.items()):
        with [col1, col2, col3][i]:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    st.success(f"✅ {service}")
                else:
                    st.error(f"❌ {service}")
            except:
                st.error(f"❌ {service}")
    
    st.divider()
    
    # Quick links
    st.subheader("🔗 Quick Links")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.link_button("📚 API Docs", f"{API_URL}/docs")
    with col2:
        st.link_button("📊 Prometheus", "http://localhost:9090")
    with col3:
        st.link_button("📈 Grafana", "http://localhost:3000")
    with col4:
        st.link_button("🎯 Metrics", f"{API_URL}/metrics")
    
    st.divider()
    
    # System commands
    st.subheader("🛠️ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared! Data refreshed.")
    
    with col2:
        if st.button("📊 Generate Reports", use_container_width=True):
            st.info("Run: `python run_pipeline.py dashboard` in terminal")

elif page == "📡 Real-Time Stream":
    st.header("📡 Real-Time Streaming Monitor")
    
    st.info("🚀 Start the streaming pipeline to see live predictions here!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Terminal 1: Start Consumer
python run_pipeline.py consumer
        """, language="bash")
    
    with col2:
        st.code("""
# Terminal 2: Start Producer
python run_pipeline.py producer \\
  --topics technology AI startups \\
  --articles 30
        """, language="bash")
    
    st.divider()
    
    # Auto-refresh latest predictions
    st.subheader("📊 Latest Predictions (Auto-refresh)")
    
    placeholder = st.empty()
    
    if st.button("Start Live Feed", type="primary"):
        for i in range(30):  # 30 iterations
            df = get_predictions_from_db()
            
            if not df.empty:
                with placeholder.container():
                    st.dataframe(
                        df.head(5)[['title', 'sentiment', 'confidence', 'processed_at']],
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            
            time.sleep(2)  # Refresh every 2 seconds

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>🚀 MLOps Sentiment Analysis Pipeline | Built with Streamlit, FastAPI, Kafka, Prometheus & Redis</p>
    <p>📊 Real-time predictions • 🎯 Model monitoring • ⚡ Production-ready</p>
</div>
""", unsafe_allow_html=True)
