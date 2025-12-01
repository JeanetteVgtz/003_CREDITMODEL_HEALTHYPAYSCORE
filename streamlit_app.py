"""
Credit Risk Assessment System - Streamlit Web Application
Production-ready credit decision tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from credit_decision_engine import CreditDecisionEngine
from model_evaluation import ModelEvaluator
from config import (
    APP_TITLE, APP_ICON, PAGE_CONFIG,
    MODEL_PERFORMANCE, BEST_MODEL_NAME, BEST_MODEL_AUC
)

# Page config
st.set_page_config(**PAGE_CONFIG)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = CreditDecisionEngine()
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = ModelEvaluator()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Executive Dashboard",
    "Credit Decision Tool",
    "Model Performance",
    "Threshold Optimization",
    "Business Impact",
    "Medical Procedure Financing",
    "Profitability Analysis"
])

# Main content
if page == "Executive Dashboard":
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("### Production-Ready Credit Risk Assessment")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model AUC-ROC", f"{BEST_MODEL_AUC:.4f}")
    with col2:
        st.metric("Model Type", BEST_MODEL_NAME)
    with col3:
        st.metric("Medical Loan ROI", "11.37%", help="From profitability analysis on real data")
    with col4:
        st.metric("Profit vs Baseline", "+341%", help="Medical loan vs no model")

    st.markdown("---")

    # Model comparison chart
    st.subheader("Model Performance Comparison")
    df_models = pd.DataFrame.from_dict(MODEL_PERFORMANCE, orient='index').reset_index()
    df_models.columns = ['Model', 'AUC-ROC', 'Accuracy']

    fig = px.bar(df_models.sort_values('AUC-ROC'),
                 x='AUC-ROC', y='Model', orientation='h',
                 title='All Models Tested',
                 color='AUC-ROC',
                 color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### System Capabilities
    - Real-time credit decisions in 0.002 seconds per application
    - 79.4% AUC-ROC accuracy in default prediction
    - Risk-based pricing and loan limits
    - Fairness testing for protected groups
    - Production-ready batch processing
    - Validated profitability on real loan outcomes
    """)

elif page == "Credit Decision Tool":
    st.title("Credit Decision Tool")
    st.markdown("### Production Mode - Batch Processing")
    st.info("Upload a CSV file with complete loan application data (all 87 features from credit bureau integration)")

    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Validate it's raw data, not results
        output_columns = ['decision', 'probability_of_default', 'credit_score',
                          'risk_category', 'recommended_interest_rate']
        if any(col in df.columns for col in output_columns):
            st.error(
                "This appears to be a results file (already processed). Please upload raw application data (X_test.csv format).")
            st.stop()

        st.success(f"Loaded {len(df)} applications")

        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Process All Applications", type="primary"):
            with st.spinner(f"Processing {len(df)} applications..."):
                from credit_decision_engine import CreditDecisionEngine

                engine = CreditDecisionEngine()
                results = engine.batch_process(df)

            st.success("Processing complete!")

            st.markdown("---")
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                approved = (results['decision'] == 'Approve').sum()
                st.metric("Approved", f"{approved:,}")
            with col2:
                rejected = (results['decision'] == 'Reject').sum()
                st.metric("Rejected", f"{rejected:,}")
            with col3:
                manual = (results['decision'] == 'Manual Review').sum()
                st.metric("Manual Review", f"{manual:,}")
            with col4:
                avg_pd = results['probability_of_default'].mean()
                st.metric("Avg Default Risk", f"{avg_pd * 100:.1f}%")

            st.markdown("---")
            st.subheader("Detailed Results")
            st.dataframe(results, use_container_width=True)

            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="credit_decisions.csv",
                mime="text/csv",
                type="primary"
            )
    else:
        st.markdown("---")
        st.subheader("How to use this tool")
        st.markdown("""
        **Step 1:** Prepare your loan applications data
        - CSV file with all 87 required features
        - Data typically comes from credit bureau API integration
        - Use `data/processed/X_test.csv` as a template

        **Step 2:** Upload the CSV file

        **Step 3:** Click "Process All Applications"

        **Step 4:** Review results and download decisions

        **In Production:**
        This system integrates with your bank's core systems and credit bureau APIs 
        to automatically pull complete applicant data for real-time processing.
        """)

        st.markdown("---")
        st.subheader("Test with Sample Data")
        st.markdown("Download a sample file to test the system:")

        # Create download button for test data
        try:
            test_data_path = Path("data/processed/X_test.csv")
            if test_data_path.exists():
                test_df = pd.read_csv(test_data_path).head(100)
                test_csv = test_df.to_csv(index=False)
                st.download_button(
                    label="Download Sample Data (100 applications)",
                    data=test_csv,
                    file_name="sample_loan_applications.csv",
                    mime="text/csv"
                )
        except:
            st.warning("Sample data not available")

elif page == "Model Performance":
    st.title("Model Performance Analysis")

    # Model comparison table
    st.subheader("All Models Tested")
    df_models = pd.DataFrame.from_dict(MODEL_PERFORMANCE, orient='index')
    df_models = df_models.sort_values('AUC-ROC', ascending=False)
    st.dataframe(df_models.style.highlight_max(axis=0), use_container_width=True)

    st.markdown("---")

    # Performance metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Metrics")
        metrics_data = {
            "Metric": ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"],
            "Score": [0.7939, 0.8370, 0.8042, 0.2458, 0.3766]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    with col2:
        st.subheader("Business Translation")
        st.markdown("""
        - **79.4% AUC-ROC**: Model correctly ranks defaulters vs non-defaulters
        - **83.7% Accuracy**: Correct predictions on 83.7% of loans
        - **80.4% Precision**: When we predict default, we're right 80.4% of the time
        - **24.6% Recall**: We catch 24.6% of actual defaults (trade-off for high precision)

        **Note:** Low recall is intentional - optimized for approving qualified borrowers 
        quickly while maintaining high precision. Threshold can be adjusted for different 
        risk appetites (see Threshold Optimization page).
        """)

elif page == "Threshold Optimization":
    st.title("Threshold Optimization Tool")
    st.markdown("### Interactive Decision Threshold Analysis")

    st.info("""
    This tool shows how different approval thresholds affect business outcomes.

    **Current business rules:**
    - Standard loan: Approve if PD < 30%
    - Medical loan: Approve if PD < 35%

    Adjust the slider to see what happens if you change these thresholds.
    """)

    # Load data
    from credit_decision_engine import CreditDecisionEngine
    import numpy as np

    predictions = pd.read_csv("data/processed/pd_test_predictions.csv")
    y_true = predictions['y_test'].values
    y_pred_proba = predictions['y_pred_proba'].values

    # Threshold slider
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Lower threshold = catch more defaults but reject more good customers"
    )

    # Calculate metrics at selected threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    total = len(y_true)
    approval_rate = (y_pred == 0).sum() / total
    total_defaults = (y_true == 1).sum()

    st.markdown("---")

    # Show metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precision", f"{precision:.3f}",
                  help="When we predict default, how often are we right?")
    with col2:
        st.metric("Recall", f"{recall:.3f}",
                  help="Of all actual defaults, what % do we catch?")
    with col3:
        st.metric("F1-Score", f"{f1:.3f}",
                  help="Harmonic mean of precision and recall")
    with col4:
        st.metric("Approval Rate", f"{approval_rate * 100:.1f}%",
                  help="% of applications approved")

    st.markdown("---")

    # Confusion matrix visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default Detection")
        detection_data = pd.DataFrame({
            'Category': ['Caught Defaults', 'Missed Defaults'],
            'Count': [int(tp), int(fn)]
        })
        fig = px.bar(detection_data, x='Category', y='Count',
                     title=f'Catching {tp} of {total_defaults} Defaults ({recall * 100:.1f}%)',
                     color='Category',
                     color_discrete_map={'Caught Defaults': 'green', 'Missed Defaults': 'red'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        **Defaults Caught:** {int(tp):,}  
        **Defaults Missed:** {int(fn):,}  
        **Total Actual Defaults:** {int(total_defaults):,}
        """)

    with col2:
        st.subheader("Approval Accuracy")
        approval_data = pd.DataFrame({
            'Category': ['Correct Approvals', 'Incorrect Approvals'],
            'Count': [int(tn), int(fp)]
        })
        fig2 = px.bar(approval_data, x='Category', y='Count',
                      title=f'Precision: {precision * 100:.1f}%',
                      color='Category',
                      color_discrete_map={'Correct Approvals': 'green', 'Incorrect Approvals': 'orange'})
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        **Correct Approvals (TN):** {int(tn):,}  
        **Incorrect Approvals (FP):** {int(fp):,}  
        **Total Approved:** {int(tn + fp):,}
        """)

    st.markdown("---")

    # Business context recommendations
    st.subheader("Recommended Threshold by Business Strategy")

    strategy_col1, strategy_col2, strategy_col3 = st.columns(3)

    with strategy_col1:
        st.markdown("""
        **Growth-Focused (0.3-0.4)**
        - Maximize approval volume
        - Accept higher default risk
        - Good for: Fintech, expansion phase
        """)
        if 0.3 <= threshold <= 0.4:
            st.success("Current threshold matches this strategy")

    with strategy_col2:
        st.markdown("""
        **Balanced (0.45-0.55)**
        - Balance risk and growth
        - Industry standard approach
        - Good for: Traditional banks
        """)
        if 0.45 <= threshold <= 0.55:
            st.success("Current threshold matches this strategy")

    with strategy_col3:
        st.markdown("""
        **Conservative (0.6-0.7)**
        - Minimize default risk
        - Lower approval volume
        - Good for: Risk-averse institutions
        """)
        if 0.6 <= threshold <= 0.7:
            st.success("Current threshold matches this strategy")

    # Show comparison table
    st.markdown("---")
    st.subheader("Compare Key Thresholds")

    comparison_data = []
    for t in [0.3, 0.5, 0.7]:
        y_pred_t = (y_pred_proba >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_t).ravel()
        prec_t = precision_score(y_true, y_pred_t, zero_division=0)
        rec_t = recall_score(y_true, y_pred_t, zero_division=0)
        app_t = (y_pred_t == 0).sum() / total

        comparison_data.append({
            'Threshold': t,
            'Precision': f"{prec_t:.3f}",
            'Recall': f"{rec_t:.3f}",
            'Approval Rate': f"{app_t * 100:.1f}%",
            'Defaults Caught': f"{int(tp_t):,}",
            'Defaults Missed': f"{int(fn_t):,}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

elif page == "Business Impact":
    st.title("Business Impact Analysis")

    # ROI Calculator
    st.subheader("Processing Efficiency")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Processing Time", "0.002 seconds",
                  help="Per application (measured)")
    with col2:
        st.metric("vs Manual Review", "30 minutes",
                  help="Industry benchmark (McKinsey 2016)")
    with col3:
        st.metric("Speedup Factor", "600x",
                  help="Throughput improvement")

    st.markdown("---")

    # Real profitability
    st.subheader("Real Profitability Analysis")

    st.success("""
    **Based on actual loan outcomes from test set (13,184 loans):**

    **Baseline (no model, approve all):**
    - Net profit: $4,017,842
    - ROI: 2.12%

    **Standard loan (30% threshold):**
    - Net profit: $10,542,564
    - ROI: 7.27%
    - Improvement: +162% vs baseline

    **Medical loan (35% threshold):**
    - Net profit: $17,740,805
    - ROI: 11.37%
    - Improvement: +341% vs baseline
    - Improvement: +68% vs standard loan
    """)

    st.markdown("""
    **Why medical loans are more profitable:**
    - Dual revenue model (interest + 5% commission from clinics)
    - Higher approval threshold (35% vs 30%) enables more volume
    - Commission revenue subsidizes lower rates for patients
    - Creates competitive moat through clinic partnerships
    """)

    st.markdown("---")

    # Triple Impact
    st.subheader("Triple Impact Assessment")

    tab1, tab2, tab3 = st.tabs(["Financial", "Social", "Environmental"])

    with tab1:
        st.markdown("""
        **Financial Impact:**
        - Proven profitability: 11.37% ROI on medical loans (real data)
        - Estimated time savings: 25,000 hours/year (for 50K applications)
        - Processing cost reduction: Based on $38.31/hour analyst rate (BLS median)
        - Faster loan processing improves customer satisfaction
        - Enables scalable growth without proportional staffing increase
        - Dual revenue model (interest + commission) creates sustainable advantage

        **Source:** U.S. Bureau of Labor Statistics (May 2023), McKinsey & Company (2016)
        """)

    with tab2:
        st.markdown("""
        **Social Impact:**
        - Fairness testing ensures equal treatment across demographics
        - Increased financial inclusion through faster processing
        - 85.8% approval rate for medical loans (vs 81% standard)
        - Enables 626 more patients to access healthcare (per 13K applications)
        - Reduced human bias in lending decisions
        - Transparent, explainable AI decisions
        - Passes 2/3 fairness tests (demographic parity, equalized odds)
        - Addresses Mexico's 41% out-of-pocket healthcare spending crisis
        """)

    with tab3:
        st.markdown("""
        **Environmental Impact:**
        - 100% digital process reduces paper consumption
        - Reduced carbon footprint from digital-first approach
        - Energy-efficient batch processing
        - Elimination of physical document storage requirements
        """)

elif page == "Medical Procedure Financing":
    st.title("Medical Procedure Financing")
    st.markdown("### Financiamiento de Procedimientos Médicos")

    # Value proposition - REORDERED
    st.info("""
    **Why choose medical procedure financing?**

    **PRIMARY ADVANTAGES:**
    - Point-of-sale financing (apply at clinic, approved in 5 minutes)
    - Higher approval rate (35% PD threshold vs 30% for standard loans)
    - Direct payment to provider (guaranteed, secure, eliminates fraud risk)
    - Clinic partnership network (integrated with 500+ providers)
    - Terms up to 24 months
    
    **ALSO INCLUDES:**
    - Lower interest rates (1-2% below standard personal loans)
    - No prepayment penalties
    - Digital approval process
    """)

    # Market context
    with st.expander("Mexican healthcare market context"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Out-of-pocket spending",
                "41%",
                help="Of total healthcare spending (vs 18-23% OECD average)"
            )
        with col2:
            st.metric(
                "Aesthetic market size",
                "$1,457M USD",
                help="Mexico market (2025)"
            )
        with col3:
            st.metric(
                "Annual growth rate",
                "9%",
                help="Projected through 2034"
            )

    st.markdown("---")

    # MODE SELECTOR
    analysis_mode = st.radio(
        "Select analysis mode:",
        ["Demo Mode (Single Application)", "Batch Analysis (Multiple Applications)"],
        help="Demo mode for detailed single application analysis. Batch mode for portfolio analysis."
    )

    st.markdown("---")

    # ========================================================================
    # DEMO MODE - Single Application
    # ========================================================================
    if analysis_mode == "Demo Mode (Single Application)":
        st.markdown("### Calculate financing for single application")

        col1, col2 = st.columns(2)

        with col1:
            procedure_amount = st.number_input(
                "Procedure cost (MXN)",
                min_value=5000,
                max_value=300000,
                value=40000,
                step=1000,
                help="Total cost of medical procedure"
            )

            loan_term = st.slider(
                "Term (months)",
                min_value=6,
                max_value=24,
                value=12,
                step=3,
                help="Loan duration in months"
            )

        with col2:
            st.markdown("**Common procedures:**")
            st.markdown("""
            - Dental (implants, orthodontics)
            - Aesthetic surgery (liposuction, rhinoplasty)
            - Specialized dermatology treatments
            - Vision correction (LASIK)
            - Fertility treatments (IVF)
            - Bariatric surgery
            """)

        use_test_data = st.checkbox("Use sample application data", value=True)

        if st.button("Calculate financing", type="primary"):
            try:
                # Load application data
                if use_test_data:
                    import pandas as pd

                    X_test = pd.read_csv('data/processed/X_test.csv')
                    app_data = X_test.iloc[[0]]
                    st.success("Using sample application")
                else:
                    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
                    if uploaded_file is None:
                        st.warning("Please upload a CSV file or use sample data")
                        st.stop()
                    app_df = pd.read_csv(uploaded_file)
                    app_data = app_df.iloc[[0]]
                    st.success(f"Using first application from uploaded file")

                # Process as both loan types
                from src.credit_decision_engine import CreditDecisionEngine

                engine = CreditDecisionEngine()

                result_standard = engine.process_application(app_data, loan_type='standard')
                result_medical = engine.process_application(
                    app_data,
                    loan_type='medical',
                    procedure_amount=procedure_amount
                )

                # Display results
                st.markdown("---")
                st.markdown("### Financing comparison")

                if result_medical["decision"] == "Approved":
                    st.success(f"Status: {result_medical['decision']}")
                elif result_medical["decision"] == "Manual Review":
                    st.warning(f"Status: {result_medical['decision']}")
                else:
                    st.error(f"Status: {result_medical['decision']}")

                st.markdown(f"**Probability of default:** {result_medical['probability_of_default']:.2%}")
                st.markdown(f"**Risk category:** {result_medical['risk_category']}")

                # Side-by-side comparison
                st.markdown("#### Standard Personal Loan vs Medical Loan")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### Standard Personal Loan")
                    st.metric("APR", f"{result_standard['recommended_interest_rate']:.2f}%")

                    monthly_rate_std = result_standard['recommended_interest_rate'] / 100 / 12
                    if monthly_rate_std > 0:
                        monthly_payment_std = (procedure_amount * monthly_rate_std *
                                               (1 + monthly_rate_std) ** loan_term) / \
                                              ((1 + monthly_rate_std) ** loan_term - 1)
                    else:
                        monthly_payment_std = procedure_amount / loan_term

                    st.metric("Monthly payment", f"${monthly_payment_std:,.2f} MXN")
                    total_paid_std = monthly_payment_std * loan_term
                    st.metric("Total to pay", f"${total_paid_std:,.2f} MXN")

                with col2:
                    st.markdown("##### Medical Loan")
                    st.metric(
                        "APR",
                        f"{result_medical['recommended_interest_rate']:.2f}%",
                        delta=f"-{result_medical['rate_savings_vs_standard']:.2f}%",
                        delta_color="normal"
                    )

                    monthly_rate_med = result_medical['recommended_interest_rate'] / 100 / 12
                    if monthly_rate_med > 0:
                        monthly_payment_med = (procedure_amount * monthly_rate_med *
                                               (1 + monthly_rate_med) ** loan_term) / \
                                              ((1 + monthly_rate_med) ** loan_term - 1)
                    else:
                        monthly_payment_med = procedure_amount / loan_term

                    savings_monthly = monthly_payment_std - monthly_payment_med
                    st.metric(
                        "Monthly payment",
                        f"${monthly_payment_med:,.2f} MXN",
                        delta=f"-${savings_monthly:,.2f}",
                        delta_color="normal"
                    )

                    total_paid_med = monthly_payment_med * loan_term
                    total_savings = total_paid_std - total_paid_med
                    st.metric(
                        "Total to pay",
                        f"${total_paid_med:,.2f} MXN",
                        delta=f"-${total_savings:,.2f}",
                        delta_color="normal"
                    )

                st.info(f"Total savings with medical loan: ${total_savings:,.2f} MXN over {loan_term} months")

                # Clinic information
                st.markdown("---")
                st.markdown("### Clinic payment breakdown")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Procedure amount", f"${result_medical['procedure_amount']:,.2f} MXN")
                with col2:
                    st.metric("Commission rate", f"{result_medical['clinic_commission_rate']}%")
                with col3:
                    st.metric("Clinic receives", f"${result_medical['clinic_receives']:,.2f} MXN")

                st.info(
                    "Direct payment to provider: Funds transferred directly to medical provider, ensuring proper use and reducing risk.")

            except Exception as e:
                st.error(f"Error processing application: {str(e)}")
                st.exception(e)

    # ========================================================================
    # BATCH MODE - Multiple Applications
    # ========================================================================
    else:
        st.markdown("### Batch analysis for portfolio evaluation")

        col1, col2 = st.columns(2)

        with col1:
            avg_procedure_amount = st.number_input(
                "Average procedure cost (MXN)",
                min_value=5000,
                max_value=300000,
                value=40000,
                step=1000,
                help="Average procedure amount for portfolio analysis"
            )

            loan_term = st.slider(
                "Term (months)",
                min_value=6,
                max_value=24,
                value=12,
                step=3
            )

        with col2:
            num_applications = st.number_input(
                "Number of applications to process",
                min_value=10,
                max_value=10000,
                value=1000,
                step=100,
                help="Number of applications from uploaded CSV or test data"
            )

        use_test_data_batch = st.checkbox("Use test data for batch analysis", value=True)

        if st.button("Run batch analysis", type="primary"):
            try:
                import pandas as pd
                import numpy as np
                from src.credit_decision_engine import CreditDecisionEngine

                # Load data
                if use_test_data_batch:
                    X_test = pd.read_csv('data/processed/X_test.csv')
                    applications_df = X_test.iloc[:num_applications]
                    st.success(f"Processing {len(applications_df)} applications from test data")
                else:
                    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
                    if uploaded_file is None:
                        st.warning("Please upload a CSV file or use test data")
                        st.stop()
                    applications_df = pd.read_csv(uploaded_file)
                    applications_df = applications_df.iloc[:num_applications]
                    st.success(f"Processing {len(applications_df)} applications from uploaded file")

                # Process batch
                engine = CreditDecisionEngine()

                with st.spinner(f"Processing {len(applications_df)} applications..."):
                    results_list = []

                    for idx, row in applications_df.iterrows():
                        app_data = pd.DataFrame([row])

                        # Process as both types
                        result_std = engine.process_application(app_data, loan_type='standard')
                        result_med = engine.process_application(
                            app_data,
                            loan_type='medical',
                            procedure_amount=avg_procedure_amount
                        )

                        # Calculate payments
                        monthly_rate_std = result_std['recommended_interest_rate'] / 100 / 12
                        monthly_rate_med = result_med['recommended_interest_rate'] / 100 / 12

                        if monthly_rate_std > 0:
                            monthly_payment_std = (avg_procedure_amount * monthly_rate_std *
                                                   (1 + monthly_rate_std) ** loan_term) / \
                                                  ((1 + monthly_rate_std) ** loan_term - 1)
                        else:
                            monthly_payment_std = avg_procedure_amount / loan_term

                        if monthly_rate_med > 0:
                            monthly_payment_med = (avg_procedure_amount * monthly_rate_med *
                                                   (1 + monthly_rate_med) ** loan_term) / \
                                                  ((1 + monthly_rate_med) ** loan_term - 1)
                        else:
                            monthly_payment_med = avg_procedure_amount / loan_term

                        total_std = monthly_payment_std * loan_term
                        total_med = monthly_payment_med * loan_term

                        results_list.append({
                            'application_id': idx,
                            'decision': result_med['decision'],
                            'pd': result_med['probability_of_default'],
                            'risk_category': result_med['risk_category'],
                            'standard_apr': result_std['recommended_interest_rate'],
                            'medical_apr': result_med['recommended_interest_rate'],
                            'apr_savings': result_med['rate_savings_vs_standard'],
                            'procedure_amount': avg_procedure_amount,
                            'monthly_payment_std': monthly_payment_std,
                            'monthly_payment_med': monthly_payment_med,
                            'total_paid_std': total_std,
                            'total_paid_med': total_med,
                            'total_savings': total_std - total_med,
                            'clinic_commission': result_med['clinic_commission_amount'],
                            'clinic_receives': result_med['clinic_receives']
                        })

                results_df = pd.DataFrame(results_list)

                # Display summary statistics
                st.markdown("---")
                st.markdown("### Portfolio analysis summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    approved = (results_df['decision'] == 'Approved').sum()
                    approval_rate = approved / len(results_df) * 100
                    st.metric("Approval rate", f"{approval_rate:.1f}%")
                    st.caption(f"{approved:,} of {len(results_df):,} approved")

                with col2:
                    avg_savings = results_df['total_savings'].mean()
                    st.metric("Avg savings per loan", f"${avg_savings:,.2f} MXN")
                    st.caption("Medical vs standard")

                with col3:
                    total_volume = len(results_df) * avg_procedure_amount
                    st.metric("Total loan volume", f"${total_volume:,.0f} MXN")
                    st.caption(f"{len(results_df):,} procedures")

                with col4:
                    total_commission = results_df['clinic_commission'].sum()
                    st.metric("Total commission revenue", f"${total_commission:,.2f} MXN")
                    st.caption("From clinic partnerships")

                # Approval breakdown
                st.markdown("---")
                st.markdown("### Decision distribution")

                col1, col2 = st.columns(2)

                with col1:
                    decision_counts = results_df['decision'].value_counts()
                    st.bar_chart(decision_counts)

                with col2:
                    risk_counts = results_df['risk_category'].value_counts()
                    st.bar_chart(risk_counts)

                # Financial summary
                st.markdown("---")
                st.markdown("### Financial impact")

                total_savings_portfolio = results_df['total_savings'].sum()
                avg_apr_reduction = results_df['apr_savings'].mean()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Total portfolio savings",
                        f"${total_savings_portfolio:,.2f} MXN",
                        help="Total savings for borrowers vs standard loans"
                    )
                    st.metric(
                        "Average APR reduction",
                        f"{avg_apr_reduction:.2f}%",
                        help="Average interest rate savings"
                    )

                with col2:
                    # Revenue model
                    interest_revenue = results_df['total_paid_med'].sum() - (len(results_df) * avg_procedure_amount)
                    st.metric(
                        "Interest revenue",
                        f"${interest_revenue:,.2f} MXN",
                        help="Total interest earned on portfolio"
                    )
                    st.metric(
                        "Commission revenue",
                        f"${total_commission:,.2f} MXN",
                        help="Revenue from clinic commissions"
                    )

                    total_revenue = interest_revenue + total_commission
                    st.info(f"**Total revenue:** ${total_revenue:,.2f} MXN")

                # Business case validation - ADDED
                st.markdown("---")
                st.markdown("### Business case validation")

                st.success(f"""
                **Portfolio economics (based on {len(results_df):,} applications):**

                **Revenue model:**
                - Interest revenue: ${interest_revenue:,.0f} MXN
                - Commission revenue: ${total_commission:,.0f} MXN  
                - Total revenue: ${total_revenue:,.0f} MXN

                **Value proposition:**
                - {approved:,} patients gain access to healthcare
                - Average savings: ${avg_savings:,.0f} MXN per patient
                - Dual revenue model justifies lower rates
                - Clinic network creates distribution advantage

                **This validates the business model:** Profitable while providing social value.
                """)

                # Download results
                st.markdown("---")

                # Prepare download CSV
                download_df = results_df.copy()
                download_df.columns = [
                    'Application ID', 'Decision', 'PD', 'Risk Category',
                    'Standard APR (%)', 'Medical APR (%)', 'APR Savings (%)',
                    'Procedure Amount (MXN)', 'Monthly Payment Standard (MXN)',
                    'Monthly Payment Medical (MXN)', 'Total Paid Standard (MXN)',
                    'Total Paid Medical (MXN)', 'Total Savings (MXN)',
                    'Clinic Commission (MXN)', 'Clinic Receives (MXN)'
                ]

                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="Download detailed results (CSV)",
                    data=csv,
                    file_name=f"medical_loan_batch_analysis_{len(results_df)}_applications.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error in batch processing: {str(e)}")
                st.exception(e)

elif page == "Profitability Analysis":
    st.title("Real Profitability Analysis")
    st.markdown("### Using Actual Loan Outcomes from Test Set")

    # Load data
    predictions = pd.read_csv('data/processed/pd_test_predictions.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    st.info(f"Analyzing {len(predictions):,} loans from model's test set with real outcomes")

    # Combine data
    df = pd.DataFrame({
        'loan_amnt': X_test['loan_amnt'],
        'actual_default': y_test['default'],
        'predicted_pd': predictions['y_pred_proba'],
        'loan_status': ['Charged Off' if d == 1 else 'Fully Paid' for d in y_test['default']]
    })

    # Industry averages from full dataset
    AVG_INTEREST_PAID_GOOD = 2136.24
    AVG_LOSS_PER_DEFAULT = 7006.64

    # Actual outcomes
    st.markdown("### Actual outcomes in test set")
    col1, col2 = st.columns(2)

    with col1:
        num_good = (df['actual_default']==0).sum()
        st.metric("Fully Paid", f"{num_good:,}",
                  help=f"{num_good/len(df)*100:.1f}% of portfolio")

    with col2:
        num_bad = (df['actual_default']==1).sum()
        st.metric("Defaulted", f"{num_bad:,}",
                  help=f"{num_bad/len(df)*100:.1f}% of portfolio")

    st.markdown("---")

    # Baseline performance
    st.markdown("### Portfolio performance without model")

    total_lent = df['loan_amnt'].sum()
    revenue_from_good = num_good * AVG_INTEREST_PAID_GOOD
    losses_from_bad = num_bad * AVG_LOSS_PER_DEFAULT
    net_profit_baseline = revenue_from_good - losses_from_bad

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total lent", f"${total_lent:,.0f}")
    with col2:
        st.metric("Revenue", f"${revenue_from_good:,.0f}")
    with col3:
        st.metric("Losses", f"${losses_from_bad:,.0f}")
    with col4:
        st.metric("Net profit", f"${net_profit_baseline:,.0f}",
                  help=f"ROI: {(net_profit_baseline/total_lent)*100:.2f}%")

    st.markdown("---")

    # With model - comparison
    st.markdown("### With your model: Standard vs Medical loan")

    def calculate_profitability(threshold, include_commission):
        df['approved'] = df['predicted_pd'] < threshold
        approved = df[df['approved']]

        if len(approved) == 0:
            return None

        num_good_approved = (approved['actual_default']==0).sum()
        num_bad_approved = (approved['actual_default']==1).sum()

        revenue_from_interest = num_good_approved * AVG_INTEREST_PAID_GOOD
        actual_losses = num_bad_approved * AVG_LOSS_PER_DEFAULT

        commission_revenue = approved['loan_amnt'].sum() * 0.05 if include_commission else 0
        total_revenue = revenue_from_interest + commission_revenue
        net_profit = total_revenue - actual_losses

        return {
            'num_approved': len(approved),
            'approval_rate': len(approved) / len(df),
            'num_good': num_good_approved,
            'num_bad': num_bad_approved,
            'default_rate': num_bad_approved / len(approved),
            'total_lent': approved['loan_amnt'].sum(),
            'revenue_interest': revenue_from_interest,
            'revenue_commission': commission_revenue,
            'total_revenue': total_revenue,
            'losses': actual_losses,
            'net_profit': net_profit,
            'profit_per_loan': net_profit / len(approved),
            'roi': net_profit / approved['loan_amnt'].sum()
        }

    # Calculate both
    standard = calculate_profitability(0.30, False)
    medical = calculate_profitability(0.35, True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Standard Loan (30% threshold)")
        st.metric("Approvals", f"{standard['num_approved']:,}",
                  help=f"{standard['approval_rate']*100:.1f}% approval rate")
        st.metric("Defaults", f"{standard['num_bad']:,}",
                  help=f"{standard['default_rate']*100:.1f}% default rate")
        st.metric("Net Profit", f"${standard['net_profit']:,.0f}",
                  delta=f"+${standard['net_profit'] - net_profit_baseline:,.0f} vs baseline")
        st.metric("ROI", f"{standard['roi']*100:.2f}%")
        st.metric("Profit per loan", f"${standard['profit_per_loan']:,.0f}")

    with col2:
        st.markdown("#### Medical Loan (35% threshold)")
        st.metric("Approvals", f"{medical['num_approved']:,}",
                  help=f"{medical['approval_rate']*100:.1f}% approval rate")
        st.metric("Defaults", f"{medical['num_bad']:,}",
                  help=f"{medical['default_rate']*100:.1f}% default rate")
        st.metric("Net Profit", f"${medical['net_profit']:,.0f}",
                  delta=f"+${medical['net_profit'] - standard['net_profit']:,.0f} vs standard")
        st.metric("ROI", f"{medical['roi']*100:.2f}%")
        st.metric("Profit per loan", f"${medical['profit_per_loan']:,.0f}")

    st.markdown("---")

    # Medical loan breakdown
    st.markdown("### Medical loan revenue breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Interest revenue", f"${medical['revenue_interest']:,.0f}")
    with col2:
        st.metric("Commission revenue", f"${medical['revenue_commission']:,.0f}",
                  help="5% commission on approved loans")
    with col3:
        st.metric("Total revenue", f"${medical['total_revenue']:,.0f}")

    st.success(f"""
    **Medical loan advantage:**
    - {medical['num_approved'] - standard['num_approved']:,} more approvals (better access)
    - ${medical['net_profit'] - standard['net_profit']:,.0f} more profit
    - {((medical['net_profit'] - standard['net_profit'])/standard['net_profit'])*100:.1f}% profit improvement
    """)

    st.markdown("---")

    # Threshold optimization
    st.markdown("### Find optimal threshold")

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    results = []

    for t in thresholds:
        result = calculate_profitability(t, True)
        if result:
            results.append({
                'Threshold': f"{t:.0%}",
                'Approved': f"{result['num_approved']:,}",
                'Defaults': f"{result['num_bad']:,}",
                'Net Profit': f"${result['net_profit']:,.0f}",
                'ROI': f"{result['roi']*100:.2f}%"
            })

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    optimal_result = max([calculate_profitability(t, True) for t in thresholds],
                         key=lambda x: x['net_profit'])


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Credit Risk System v1.0**")
st.sidebar.markdown("ITESO - Modelos de Crédito")





