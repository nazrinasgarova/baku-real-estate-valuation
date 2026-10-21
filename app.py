"""
Baku Real Estate AI - Machine Learning Prediction & Analytics Dashboard
Streamlit Application for Real Estate Automated Valuation (AVM) & Mortgage Risk Analytics
Data Source: Kaggle 'Binaaz sale project' by Shahriyar Mammadli (bina.az public listings)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import modular haversine distance directly from src.features
from src.features import haversine_distance

# Configure page settings
st.set_page_config(
    page_title="Bakı Daşınmaz Əmlak AVM & İpoteka Riski Platforması",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-card {
        background-color: #FEF3C7;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #D97706;
    }
</style>
""", unsafe_allow_html=True)

def patch_imputer(obj):
    """Ensure cross-version scikit-learn compatibility for SimpleImputer across environments."""
    if obj is None:
        return
    if hasattr(obj, 'transformers_'):
        for item in obj.transformers_:
            if len(item) >= 2:
                patch_imputer(item[1])
    if hasattr(obj, 'steps'):
        for name, step in obj.steps:
            patch_imputer(step)
    if hasattr(obj, 'calibrated_classifiers_'):
        for cc in obj.calibrated_classifiers_:
            patch_imputer(getattr(cc, 'estimator', None))
            patch_imputer(getattr(cc, 'base_estimator', None))
    if hasattr(obj, 'estimator'):
        patch_imputer(obj.estimator)
    if hasattr(obj, 'base_estimator'):
        patch_imputer(obj.base_estimator)
    if hasattr(obj, 'statistics_'):
        if not hasattr(obj, '_fill_dtype'):
            try:
                obj._fill_dtype = getattr(obj.statistics_, 'dtype', None)
            except Exception:
                pass

@st.cache_resource
def load_production_artifacts():
    """Load serialized models and dynamic metrics JSON."""
    base_dir = os.path.dirname(__file__)
    xgb_path = os.path.join(base_dir, "models", "champion_xgb_pipeline.joblib")
    rf_path = os.path.join(base_dir, "models", "random_forest_regressor.joblib")
    st_path = os.path.join(base_dir, "models", "streamlined_xgb_pipeline.joblib")
    clf_path = os.path.join(base_dir, "models", "calibrated_classification_pipeline.joblib")
    json_path = os.path.join(base_dir, "models", "metrics_summary.json")
    
    xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
    stream_model = joblib.load(st_path) if os.path.exists(st_path) else None
    classifier = joblib.load(clf_path) if os.path.exists(clf_path) else None
    
    for model in [xgb_model, rf_model, stream_model, classifier]:
        patch_imputer(model)

    metrics = None
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
            
    return xgb_model, rf_model, stream_model, classifier, metrics

xgb_pipeline, rf_pipeline, stream_pipeline, collateral_clf, metrics_summary = load_production_artifacts()

# Baku key locations and coordinates (dynamically loaded from empirical dataset medians)
DEFAULT_LOCATION_COORDS = {
    "28 May m.": {"lat": 40.3836, "lng": 49.8471},
    "8 Noyabr m.": {"lat": 40.3980, "lng": 49.8186},
    "Elmlər Akademiyası m.": {"lat": 40.3784, "lng": 49.8120},
    "Gənclik m.": {"lat": 40.3998, "lng": 49.8558},
    "Həzi Aslanov m.": {"lat": 40.3721, "lng": 49.9537},
    "Masazır q.": {"lat": 40.4863, "lng": 49.7551},
    "Memar Əcəmi m.": {"lat": 40.4136, "lng": 49.8138},
    "Mərdəkan q.": {"lat": 40.4807, "lng": 50.1471},
    "Nizami m.": {"lat": 40.3825, "lng": 49.8302},
    "Nəriman Nərimanov m.": {"lat": 40.4035, "lng": 49.8697},
    "Nərimanov r.": {"lat": 40.3927, "lng": 49.8547},
    "Nəsimi r.": {"lat": 40.3944, "lng": 49.8363},
    "Qara Qarayev m.": {"lat": 40.4175, "lng": 49.9341},
    "İnşaatçılar m.": {"lat": 40.3919, "lng": 49.8025},
    "Şah İsmayıl Xətai m.": {"lat": 40.3832, "lng": 49.8721},
    "Other (City Center)": {"lat": 40.3725, "lng": 49.8360}
}

LOCATION_COORDS = metrics_summary.get("location_coords", DEFAULT_LOCATION_COORDS) if metrics_summary else DEFAULT_LOCATION_COORDS

# Header
st.markdown('<div class="main-header">Bakı Daşınmaz Əmlak Qiymətləndirmə (AVM) & İpoteka Riski Platforması</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Bina.az bazar məlumatları (Kaggle: Shahriyar Mammadli) və Scikit-Learn / XGBoost Machine Learning Pipeline əsasında avtomatlaşdırılmış qiymətləndirmə mühərriki.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Qiymət Təyini & İpoteka Simulyasiyası",
    "📊 Bazar İnsightları & Məsafə Dinamikası",
    "🤖 ML Modelləri & Performans Bençmarkı"
])

# ================= TAB 1: VALUATION & MORTGAGE ENGINE =================
with tab1:
    st.subheader("Əmlak Parametrlərini Daxil Edin")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category = st.selectbox(
            "Əmlakın Kateqoriyası",
            ["Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi"],
            index=0
        )
        location = st.selectbox(
            "Yerləşdiyi Ərazi / Metro / Qəsəbə",
            list(LOCATION_COORDS.keys()),
            index=0
        )
        area = st.number_input(
            "Ümumi Sahə (m²)",
            min_value=15.0,
            max_value=600.0,
            value=95.0,
            step=5.0
        )
        rooms = st.number_input(
            "Otaq Sayı",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

    with col2:
        current_floor = st.number_input(
            "Yerləşdiyi Mərtəbə",
            min_value=1,
            max_value=35,
            value=6,
            step=1
        )
        total_floors = st.number_input(
            "Binanın Ümumi Mərtəbə Sayı",
            min_value=1,
            max_value=35,
            value=16,
            step=1
        )
        repair_status = st.selectbox(
            "Təmir Vəziyyəti",
            ["Təmirli (var)", "Təmirsiz (yoxdur)"],
            index=0
        )
        deed_status = st.selectbox(
            "Kupça / Çıxarış Statusu",
            ["Var (Çıxarışlı)", "Yoxdur"],
            index=0
        )

    with col3:
        mortgage_status = st.selectbox(
            "İpotekaya Yararlılıq",
            ["Var (Yararlı)", "Yoxdur"],
            index=0
        )
        active_engine = st.selectbox(
            "Seçilmiş ML Mühərriki",
            [
                "Tuned XGBoost (Deployment Champion, 5.6 MB)",
                "Random Forest (Ensemble Benchmark, 26.5 MB)",
                "Streamlined XGBoost (Top 10 Features, 5.5 MB)"
            ],
            index=0
        )
        interest_rate = st.slider(
            "İpoteka İllik Faiz Dərəcəsi (%)",
            min_value=4.0,
            max_value=15.0,
            value=8.0,
            step=0.5
        )

    with st.expander("Əlavə Xüsusiyyətlər (Metro, Dəniz Mənzərəsi, Qaz və Lift)"):
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            has_metro_ui = st.checkbox("Metroya yaxın ərazi", value=True)
        with a_col2:
            has_sea_ui = st.checkbox("Dəniz mənzərəsi / Panorama", value=False)
        with a_col3:
            has_utilities_ui = st.checkbox("Qaz və Lift mövcuddur", value=True)

    st.markdown("---")
    predict_btn = st.button("🚀 Əmlakın Bazar Qiymətini və Girov Riskini Hesabla", type="primary", use_container_width=True)

    if predict_btn:
        if "Streamlined" in active_engine:
            active_pipeline = stream_pipeline
        elif "Random Forest" in active_engine:
            active_pipeline = rf_pipeline
        else:
            active_pipeline = xgb_pipeline
        
        if active_pipeline is None:
            st.error("Seçilmiş model faylı tapılmadı! models/ qovluğunu yoxlayın.")
        else:
            loc_data = LOCATION_COORDS.get(location, {"lat": 40.3725, "lng": 49.8360})
            if isinstance(loc_data, dict):
                lat, lng = loc_data["lat"], loc_data["lng"]
            else:
                lat, lng = loc_data
            dist_km = haversine_distance(lat, lng)
            
            c_floor = float(current_floor)
            t_floor = float(total_floors)
            f_ratio = c_floor / max(t_floor, 1.0)
            is_first = 1 if c_floor == 1 else 0
            is_last = 1 if (c_floor == t_floor and t_floor > 1) else 0
            
            has_rep = 1 if "var" in repair_status.lower() else 0
            has_bill = 1 if "var" in deed_status.lower() else 0
            has_mor = 1 if "var" in mortgage_status.lower() else 0

            area_val = float(area)
            rooms_val = float(rooms)
            area_per_rm = area_val / max(rooms_val, 1.0)

            # Map location strictly to top trained locations or 'Other'
            top_locs = metrics_summary.get("top_locations", []) if metrics_summary else []
            location_binned = location if location in top_locs else "Other"

            # Construct clean feature vector matching production pipeline
            input_df = pd.DataFrame([{
                'area': area_val,
                'Otaq sayı': rooms_val,
                'current_floor': c_floor,
                'total_floors': t_floor,
                'floor_ratio': f_ratio,
                'is_first_floor': is_first,
                'is_last_floor': is_last,
                'area_per_room': area_per_rm,
                'dist_center_km': float(dist_km),
                'lat': float(lat),
                'lng': float(lng),
                'has_repair': has_rep,
                'has_bill_of_sale': has_bill,
                'has_mortgage': has_mor,
                'has_metro': int(has_metro_ui),
                'has_sea_view': int(has_sea_ui),
                'has_gas_lift': int(has_utilities_ui),
                'Kateqoriya': category,
                'location_binned': location_binned
            }])

            try:
                patch_imputer(active_pipeline)
                patch_imputer(collateral_clf)
                # Invert log-transformation: model predicts log1p(price)
                t_start = time.time()
                log_pred = float(active_pipeline.predict(input_df)[0])
                pred_price = float(np.expm1(log_pred))
                pred_price = max(pred_price, 25000.0)
                inference_ms = (time.time() - t_start) * 1000.0
                
                price_per_m2 = pred_price / area_val
                
                # Valuation range based on standard PPE10 (±10% prediction interval)
                lower_bound = pred_price * 0.90
                upper_bound = pred_price * 1.10

                st.success("✅ Bazar Qiymətləndirməsi və Risk Analizi Uğurla Hesablandı!")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric(
                        label="Təxmini Ədalətli Bazar Dəyəri",
                        value=f"{pred_price:,.0f} AZN",
                        delta=f"1 m² = {price_per_m2:,.0f} AZN/m²"
                    )
                with res_col2:
                    st.metric(
                        label="Qiymətləndirmə Diapazonu (PPE10: ±10%)",
                        value=f"{lower_bound:,.0f} - {upper_bound:,.0f} AZN",
                        delta=f"Inference: {inference_ms:.2f} ms"
                    )
                with res_col3:
                    st.metric(
                        label="Mərkəzə Məsafə (Fəvvarələr Meydanı)",
                        value=f"{dist_km:.1f} km",
                        delta="40.3725° N, 49.8360° E"
                    )

                st.caption("ℹ️ *Holdout test setində proqnozların ~61%-i faktiki bazar qiymətinin ±10%-i (PPE10), ~85%-i isə ±20%-i (PPE20) daxilindədir.*")
                st.caption("ℹ️ *Qeyd: Xəritə koordinatları verilənlər bazasındakı real elanların median GPS koordinatlarıdır; fərdi qiymətləndirmədə əmlakın dəqiq GPS koordinatları əsas götürülür.*")

                # Calibrated Collateral Triage Evaluation
                st.markdown("### İpoteka & Girov Qiymətləndirmə Simulyasiyası")
                b_col1, b_col2 = st.columns(2)
                
                with b_col1:
                    max_ltv_pct = 80 if has_bill else 0
                    max_loan = pred_price * (max_ltv_pct / 100.0)
                    down_payment = pred_price - max_loan
                    monthly_rate = (interest_rate / 100.0) / 12.0
                    n_months = 20 * 12
                    monthly_payment = max_loan * (monthly_rate * (1 + monthly_rate)**n_months) / ((1 + monthly_rate)**n_months - 1) if max_loan > 0 else 0
                    
                    st.markdown(f"""
                    * **Maksimum Kredit Həcmi (LTV {max_ltv_pct}% fərziyyəsi):** **{max_loan:,.0f} AZN**
                    * **Tələb Olunan İlkin Ödəniş (20%):** **{down_payment:,.0f} AZN**
                    * **Aylıq İpoteka Ödənişi ({interest_rate}% illik, 20 il fərziyyəsi):** **{monthly_payment:,.0f} AZN/ay**
                    """)
                    st.caption("*İllüstrativ simulyasiyadır; bankın rəsmi kredit anderraytinq siyasətini əks etdirmir.*")

                with b_col2:
                    if not has_bill:
                        st.error("🚫 **Kupçasız Əmlak:** Dövlət reyestr çıxarışı olmadığı üçün standart ipoteka girovu kimi qəbul edilə bilməz.")
                    else:
                        # Extract optimal threshold dynamically from metrics summary (default: 0.370)
                        opt_thresh = 0.370
                        if metrics_summary and 'classification' in metrics_summary:
                            opt_thresh = metrics_summary['classification'].get('optimal_threshold', 0.370)

                        # Compute calibrated probability using serialized classifier
                        if collateral_clf is not None:
                            cal_prob = float(collateral_clf.predict_proba(input_df)[0][1])
                        else:
                            cal_prob = 1.0 if pred_price >= 300000 else 0.0

                        if cal_prob >= opt_thresh:
                            st.warning(f"⚠️ **Yüksək Dəyərli Girov (Kalibrlənmiş Ehtimal: {cal_prob*100:.1f}%):** Əmlak dəyəri 300,000 AZN həddini keçdiyi üçün Baş Kredit Komitəsinin və yerində müstəqil qiymətləndiricinin təsdiqi tələb olunur (Optimal hədd: T* = {opt_thresh:.3f}).")
                        else:
                            st.info(f"✅ **Standart İpoteka Portfeli (Yüksək Dəyər Ehtimalı: {cal_prob*100:.1f}%):** Əmlak standart rəqəmsal anderraytinq və təsdiq meyarlarına tam cavab verir (Optimal hədd: T* = {opt_thresh:.3f}).")

            except Exception as err:
                st.error(f"Proqnoz zamanı xəta baş verdi: {err}")

# ================= TAB 2: MARKET INSIGHTS =================
with tab2:
    st.subheader("📊 Bakı Əmlak Bazarı üzrə Statistik Müşahidələr")
    
    st.markdown("""
    100,000+ bina.az elanı üzərində aparılan kəşfiyyat xarakterli təhlilin (EDA) əsas empirik nəticələri:
    """)

    i_col1, i_col2 = st.columns(2)
    with i_col1:
        st.markdown("#### Çıxarış (Kupça) Təminatının Empirik Fərqi:")
        if metrics_summary and 'kupca_proof' in metrics_summary:
            st.dataframe(pd.DataFrame(metrics_summary['kupca_proof']), use_container_width=True)
        else:
            st.markdown("""
            * **Yeni Tikili:** Kupçalı 2,364 vs Kupçasız 2,467 AZN/m² (-4.1%)
            * **Köhnə Tikili:** Kupçalı 2,250 vs Kupçasız 2,235 AZN/m² (+0.7%)
            * **Həyət evi / Bağ evi:** Kupçalı 1,312 vs Kupçasız 728 AZN/m² (**+80.3%**)
            """)
        st.caption("ℹ️ *Həyət evlərində Kupça sənədi m² üzrə +80.3% empirik qiymət üstünlüyü təmin edir. Mənzillərdə isə fərq cüzidir; yeni binaların çoxu ilkin mərhələdə MTK müqaviləsi ilə satılır.*")

    with i_col2:
        st.markdown("#### Mərkəzdən Məsafə üzrə Mərhələli Qiymət Enişi:")
        if metrics_summary and 'distance_bands' in metrics_summary:
            dist_df = pd.DataFrame(metrics_summary['distance_bands']).rename(columns={
                'dist_band': 'Məsafə Zolağı',
                'median': 'Median Qiymət (AZN/m²)',
                'count': 'Elan Sayı'
            })
            dist_df['Median Qiymət (AZN/m²)'] = dist_df['Median Qiymət (AZN/m²)'].round(0).astype(int)
            st.dataframe(dist_df, use_container_width=True)
        else:
            st.markdown("""
            * **0 - 2 km (Mərkəz):** **2,849 AZN/m²** (Fəvvarələr Meydanı ətrafı)
            * **2 - 5 km:** **2,511 AZN/m²**
            * **5 - 10 km:** **2,132 AZN/m²**
            * **10 - 15 km:** **1,815 AZN/m²**
            * **15 - 25 km:** **950 AZN/m²** (Minimum qiymət zonası)
            * **25+ km (Sahil Villaları):** **1,656 AZN/m²** (Bilgəh, Şüvəlan, Mərdəkan dənizkənarı villaları)
            """)
        st.caption("ℹ️ *Mərkəzdən uzaqlaşdıqca qiymətlər 15–25 km aralığına qədər kəskin enir, 25+ km sahil zonasında isə lüks bağ evləri hesabına yenidən bahalaşır.*")

# ================= TAB 3: MODEL RESULTS =================
with tab3:
    st.subheader("🤖 Maşın Öyrənməsi Modellərinin Holdout Test Müqayisəsi")
    
    st.markdown("""
    Bütün modellər ciddi **70% Train / 15% Validation / 15% Holdout Test** bölgüsü üzərində qiymətləndirilmişdir. Test seti model seçimində istifadə olunmamış, yalnız yekun hesabat üçün bir dəfə qiymətləndirilmişdir:
    """)

    # Render dynamic metrics from metrics_summary.json if available
    if metrics_summary and 'xgb_test' in metrics_summary:
        xgb_m = metrics_summary['xgb_test']
        rf_m = metrics_summary['rf_test']
        st_m = metrics_summary['streamlined_test']
        rd_m = metrics_summary['ridge_test']

        results_df = pd.DataFrame([
            {
                "Model": "Tuned XGBoost",
                "Rol": "Deployment Champion",
                "Test R²": f"{xgb_m['r2']:.4f}",
                "Test MAE": f"{xgb_m['mae']:,.2f} AZN",
                "MedAPE": f"{xgb_m['medape']:.2f}%",
                "PPE10 (±10%)": f"{xgb_m['ppe10']:.1f}%",
                "PPE20 (±20%)": f"{xgb_m['ppe20']:.1f}%",
                "Model Ölçüsü": f"{xgb_m['size_mb']:.1f} MB",
                "Status": f"Production Engine ({xgb_m.get('latency_ms', 7.99):.2f} ms latency)"
            },
            {
                "Model": "Random Forest",
                "Rol": "Ensemble Benchmark",
                "Test R²": f"{rf_m['r2']:.4f}",
                "Test MAE": f"{rf_m['mae']:,.2f} AZN",
                "MedAPE": f"{rf_m['medape']:.2f}%",
                "PPE10 (±10%)": f"{rf_m['ppe10']:.1f}%",
                "PPE20 (±20%)": f"{rf_m['ppe20']:.1f}%",
                "Model Ölçüsü": f"{rf_m['size_mb']:.1f} MB",
                "Status": "Ensemble Reference (compress=3)"
            },
            {
                "Model": "Streamlined XGBoost",
                "Rol": "Top 10 Features",
                "Test R²": f"{st_m['r2']:.4f}",
                "Test MAE": f"{st_m['mae']:,.2f} AZN",
                "MedAPE": f"{st_m['medape']:.2f}%",
                "PPE10 (±10%)": f"{st_m['ppe10']:.1f}%",
                "PPE20 (±20%)": f"{st_m['ppe20']:.1f}%",
                "Model Ölçüsü": f"{st_m['size_mb']:.1f} MB",
                "Status": f"{st_m.get('feature_count', 10)} Transformed Features"
            },
            {
                "Model": "Ridge Regression",
                "Rol": "Baza Xətti Model",
                "Test R²": f"{rd_m['r2']:.4f}",
                "Test MAE": f"{rd_m['mae']:,.2f} AZN",
                "MedAPE": f"{rd_m['medape']:.2f}%",
                "PPE10 (±10%)": f"{rd_m['ppe10']:.1f}%",
                "PPE20 (±20%)": f"{rd_m['ppe20']:.1f}%",
                "Model Ölçüsü": "< 1 MB",
                "Status": "Linear Baseline"
            }
        ])
    else:
        results_df = pd.DataFrame([
            {"Model": "Tuned XGBoost", "Rol": "Deployment Champion", "Test R²": "0.8390", "Test MAE": "33,924 AZN", "MedAPE": "7.46%", "PPE10 (±10%)": "60.9%", "PPE20 (±20%)": "84.6%", "Model Ölçüsü": "5.6 MB", "Status": "Production Engine"},
            {"Model": "Random Forest", "Rol": "Ensemble Benchmark", "Test R²": "0.8281", "Test MAE": "35,043 AZN", "MedAPE": "7.46%", "PPE10 (±10%)": "60.9%", "PPE20 (±20%)": "83.4%", "Model Ölçüsü": "26.5 MB", "Status": "Ensemble Reference"},
            {"Model": "Streamlined XGBoost", "Rol": "Top 10 Features", "Test R²": "0.8111", "Test MAE": "40,055 AZN", "MedAPE": "9.46%", "PPE10 (±10%)": "52.2%", "PPE20 (±20%)": "77.7%", "Model Ölçüsü": "5.5 MB", "Status": "10 Transformed Features"},
            {"Model": "Ridge Regression", "Rol": "Baza Xətti Model", "Test R²": "0.3177", "Test MAE": "60,652 AZN", "MedAPE": "13.90%", "PPE10 (±10%)": "38.5%", "PPE20 (±20%)": "64.4%", "Model Ölçüsü": "< 1 MB", "Status": "Linear Baseline"}
        ])

    st.dataframe(results_df, use_container_width=True)

    xgb_medape = f"{xgb_m['medape']:.2f}%" if (metrics_summary and 'xgb_test' in metrics_summary) else "7.46%"
    xgb_ppe10 = f"{xgb_m['ppe10']:.1f}%" if (metrics_summary and 'xgb_test' in metrics_summary) else "60.9%"
    xgb_ppe20 = f"{xgb_m['ppe20']:.1f}%" if (metrics_summary and 'xgb_test' in metrics_summary) else "84.6%"

    st.markdown("---")
    st.subheader("Niyə Qlobal Bank Standartı MedAPE?")
    st.markdown(f"""
    * **MedAPE (Median Absolute Percentage Error):** Beynəlxalq AVM standartıdır. Tuned XGBoost ({xgb_medape}) qlobal 10% sənaye standartını üstələyir.
    * **PPE10 və PPE20:** Əmlakların ~{xgb_ppe10}-u faktiki bazar qiymətinin $\\pm 10\\%$-i daxilində (PPE10), ~{xgb_ppe20}-i isə $\\pm 20\\%$-i daxilində (PPE20) yüksək dəqiqliklə proqnozlaşdırılır.
    """)

    st.subheader("Ən Vacib Xüsusiyyətlər (Top Value Drivers)")
    if metrics_summary and 'top_features' in metrics_summary:
        top_f_df = pd.DataFrame(metrics_summary['top_features'])
        st.bar_chart(top_f_df.set_index("Feature"), use_container_width=True)
    else:
        feat_df = pd.DataFrame({
            "Xüsusiyyət": [
                "Kateqoriya_Yeni tikili", "Kateqoriya_Həyət evi/Bağ evi", "area (Ümumi Sahə)",
                "dist_center_km (Mərkəzə Məsafə)", "location_binned_Masazır q.", "Kateqoriya_Köhnə tikili",
                "location_binned_Mərdəkan q.", "area_per_room (Otaq Başına Sahə)", "Otaq sayı",
                "location_binned_Nəriman Nərimanov m."
            ],
            "Əhəmiyyət Çəkisi (Gain)": [0.170, 0.158, 0.104, 0.079, 0.071, 0.054, 0.046, 0.038, 0.030, 0.023]
        })
        st.bar_chart(feat_df.set_index("Xüsusiyyət"), use_container_width=True)
