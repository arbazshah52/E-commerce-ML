"""Simple Streamlit app for E-commerce ML order prediction."""

import streamlit as st
from src.api import load_model_package, predict_session, get_model_info


def main():
    st.set_page_config(page_title="E-commerce Order Predictor", layout="wide")
    
    st.title("🛍️ E-commerce Order Prediction")
    st.markdown("Predict whether a customer session will result in an order based on their behavior")
    
    # Load model once (cached)
    @st.cache_resource
    def get_model():
        return load_model_package()
    
    try:
        model_package = get_model()
        
        # Main prediction interface
        st.subheader("Session Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_clicks = st.slider(
                "Number of Clicks",
                min_value=0,
                max_value=100,
                value=15,
                help="Total number of product clicks in the session"
            )
            num_carts = st.slider(
                "Items Added to Cart",
                min_value=0,
                max_value=50,
                value=3,
                help="Number of times items were added to cart"
            )
        
        with col2:
            num_events = st.slider(
                "Total Events",
                min_value=0,
                max_value=200,
                value=20,
                help="Total number of events in the session"
            )
            num_unique_items = st.slider(
                "Unique Items Viewed",
                min_value=0,
                max_value=100,
                value=8,
                help="Number of different products viewed"
            )
        
        # Make prediction
        session_data = {
            "num_clicks": num_clicks,
            "num_carts": num_carts,
            "num_events": num_events,
            "num_unique_items": num_unique_items,
        }
        
        result = predict_session(session_data, model_package)
        
        # Display results
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result["prediction"] == 1:
                st.success("✅ Order Likely")
            else:
                st.warning("⚠️ No Order Expected")
        
        with col2:
            st.metric(
                "Order Probability",
                f"{result['probability']:.1%}",
                delta=None
            )
        
        with col3:
            st.info(f"Model: {result['model_name']}")
        
        # Model information
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Metrics")
            info = get_model_info()
            metrics = info.get("test_metrics", {})
            
            if metrics:
                metric_cols = st.columns(3)
                metrics_list = [
                    ("Accuracy", "accuracy"),
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1-Score", "f1_score"),
                    ("ROC-AUC", "roc_auc"),
                ]
                
                for i, (label, key) in enumerate(metrics_list):
                    col = metric_cols[i % 3]
                    value = metrics.get(key, 0)
                    col.metric(label, f"{value:.2%}")
        
        with col2:
            st.subheader("ℹ️ About This Model")
            with st.expander("Model Details", expanded=False):
                st.json(info)
                st.markdown("""
                **Features Used:**
                - **num_clicks**: Count of click events in the session
                - **num_carts**: Count of cart events in the session  
                - **num_events**: Total events in the session
                - **num_unique_items**: Unique items viewed in the session
                
                **Model Type:** Logistic Regression (with scaling)
                """)
        
        # Session data display
        with st.expander("📋 Session Data", expanded=False):
            st.write("**Input Features:**")
            for key, value in session_data.items():
                st.write(f"- {key}: {value}")
        
    except FileNotFoundError:
        st.error(
            "❌ Model not found! Please run `python src/train.py` to train the model first."
        )
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
