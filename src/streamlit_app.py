
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main
from pathlib import Path
import mlflow


def load_mlflow_results():
    """Load results from MLflow runs."""
    mlruns_path = Path("mlruns/0")
    results = []
    
    if mlruns_path.exists():
        for run_dir in mlruns_path.glob("*"):
            if run_dir.is_dir() and run_dir.name != "meta.yaml":
                run_data = {"run_id": run_dir.name}
                
                # Load metrics
                metrics_path = run_dir / "metrics"
                if metrics_path.exists():
                    for metric_file in metrics_path.glob("*"):
                        try:
                            with open(metric_file, 'r') as f:
                                value = float(f.read().strip())
                                run_data[metric_file.name] = value
                        except:
                            pass
                
                # Load parameters
                params_path = run_dir / "params"
                if params_path.exists():
                    for param_file in params_path.glob("*"):
                        try:
                            with open(param_file, 'r') as f:
                                value = f.read().strip()
                                run_data[f"param_{param_file.name}"] = value
                        except:
                            pass
                
                results.append(run_data)
    
    return results


def load_image_safely(image_path, caption="Image"):
    """Safely load and display an image, with fallback if file doesn't exist."""
    try:
        if Path(image_path).exists():
            image = Image.open(image_path)
            st.image(image, caption=caption)
            return True
        else:
            st.warning(f"Image not found: {image_path}")
            st.info("This image would show the pipeline overview. Please ensure the image file exists in the _assets folder.")
            return False
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return False


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Pipeline Overview", "Prediction", "Results & MLflow"])

    with tab1:
        # Try to load high level overview image
        high_level_loaded = load_image_safely("_assets/high_level_overview.png", "High Level Pipeline")

        # Try to load whole pipeline image
        whole_pipeline_loaded = load_image_safely("_assets/training_and_deployment_pipeline_updated.png", "Whole Pipeline")

        st.markdown(
            """ 
        #### Problem Statement 
         The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
        )
        
        if not whole_pipeline_loaded:
            st.markdown(
                """ 
            Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
            """
            )

        st.markdown(
            """ 
        #### Description of Features 
        This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
        | Models        | Description   | 
        | ------------- | -     | 
        | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
        | Payment Installments   | Number of installments chosen by the customer. |  
        | Payment Value |       Total amount paid by the customer. | 
        | Price |       Price of the product. |
        | Freight Value |    Freight value of the product.  | 
        | Product Name length |    Length of the product name. |
        | Product Description length |    Length of the product description. |
        | Product photos Quantity |    Number of product published photos |
        | Product weight measured in grams |    Weight of the product measured in grams. | 
        | Product length (CMs) |    Length of the product measured in centimeters. |
        | Product height (CMs) |    Height of the product measured in centimeters. |
        | Product width (CMs) |    Width of the product measured in centimeters. |
        """
        )

    with tab2:
        st.header("Make Predictions")
        payment_sequential = st.sidebar.slider("Payment Sequential", 1, 10, 1)
        payment_installments = st.sidebar.slider("Payment Installments", 1, 24, 1)
        payment_value = st.number_input("Payment Value", min_value=0.0, value=100.0)
        price = st.number_input("Price", min_value=0.0, value=50.0)
        freight_value = st.number_input("freight_value", min_value=0.0, value=10.0)
        product_name_length = st.number_input("Product name length", min_value=1, value=50)
        product_description_length = st.number_input("Product Description length", min_value=1, value=200)
        product_photos_qty = st.number_input("Product photos Quantity ", min_value=0, value=1)
        product_weight_g = st.number_input("Product weight measured in grams", min_value=0.0, value=500.0)
        product_length_cm = st.number_input("Product length (CMs)", min_value=0.0, value=20.0)
        product_height_cm = st.number_input("Product height (CMs)", min_value=0.0, value=10.0)
        product_width_cm = st.number_input("Product width (CMs)", min_value=0.0, value=15.0)

        if st.button("Predict"):
            try:
                service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                running=False,
                )
                if service is None:
                    st.write(
                        "No service could be found. The pipeline will be run first to create a service."
                    )
                    st.info("Please run the deployment pipeline first using: python src/run_deployment.py")
                    return

                df = pd.DataFrame(
                    {
                        "payment_sequential": [payment_sequential],
                        "payment_installments": [payment_installments],
                        "payment_value": [payment_value],
                        "price": [price],
                        "freight_value": [freight_value],
                        "product_name_lenght": [product_name_length],
                        "product_description_lenght": [product_description_length],
                        "product_photos_qty": [product_photos_qty],
                        "product_weight_g": [product_weight_g],
                        "product_length_cm": [product_length_cm],
                        "product_height_cm": [product_height_cm],
                        "product_width_cm": [product_width_cm],
                    }
                )
                json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
                data = np.array(json_list)
                pred = service.predict(data)
                st.success(
                    "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                        pred
                    )
                )
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure the deployment pipeline has been run successfully.")

    with tab3:
        st.header("Pipeline Results & MLflow")
        
        # Load and display MLflow results
        results = load_mlflow_results()
        
        if results:
            st.subheader("📊 MLflow Pipeline Runs")
            
            # Convert to DataFrame for better display
            df = pd.DataFrame(results)
            
            if not df.empty:
                # Sort by run_id (most recent first)
                df = df.sort_values('run_id', ascending=False)
                
                # Display metrics
                metrics_cols = [col for col in df.columns if col not in ['run_id'] and not col.startswith('param_')]
                
                if metrics_cols:
                    st.subheader("📈 Metrics")
                    metrics_df = df[['run_id'] + metrics_cols].copy()
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Show latest metrics prominently
                    if len(metrics_df) > 0:
                        latest = metrics_df.iloc[0]
                        st.subheader("🎯 Latest Run Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if 'r2_score' in latest:
                                st.metric("R² Score", f"{latest['r2_score']:.4f}")
                        with col2:
                            if 'rmse' in latest:
                                st.metric("RMSE", f"{latest['rmse']:.4f}")
                        with col3:
                            if 'mse' in latest:
                                st.metric("MSE", f"{latest['mse']:.4f}")
                
                # Display parameters
                param_cols = [col for col in df.columns if col.startswith('param_')]
                if param_cols:
                    st.subheader("⚙️ Model Parameters")
                    param_df = df[['run_id'] + param_cols].copy()
                    # Remove 'param_' prefix for display
                    param_df.columns = ['run_id'] + [col.replace('param_', '') for col in param_cols]
                    st.dataframe(param_df, use_container_width=True)
        else:
            st.warning("No MLflow results found. Please run the pipeline first.")
            st.info("Run: `python src/run_pipeline.py`")
        
        # Original results section
        st.subheader("Model Comparison Results")
        if st.button("Show Results"):
            st.write(
                "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
            )

            df = pd.DataFrame(
                {
                    "Models": ["LightGBM", "Xgboost"],
                    "MSE": [1.804, 1.781],
                    "RMSE": [1.343, 1.335],
                }
            )
            st.dataframe(df)

            st.write(
                "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
            )
            load_image_safely("_assets/feature_importance_gain.png", "Feature Importance Gain")
        
        # MLflow UI link
        st.subheader("🔗 MLflow UI")
        st.info("To view detailed MLflow results, run:")
        st.code("mlflow ui --backend-store-uri file:./mlruns --port 5000")
        st.info("Then open Chrome and go to: http://localhost:5000")


if __name__ == "__main__":
    main()
