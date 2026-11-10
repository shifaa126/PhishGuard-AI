
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import zxingcpp
import re

from datetime import datetime
from urllib.parse import urlparse
from feature_extraction import extract_features


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f4f2ff;
}

.block-container {
    max-width: 1200px;
    padding: 2rem 3rem;
}

section[data-testid="stSidebar"] {
    background: #121a30;
}

section[data-testid="stSidebar"] * {
    color: #eeeeff;
}

h1, h2, h3 {
    color: #dfd0ff !important;
}

p, label {
    color: #eeeeF8;
}

.stTextInput input {
    background: #202942 !important;
    color: white !important;
    border: 1px solid #586080 !important;
    border-radius: 10px;
}

.stButton button, .stDownloadButton button {
    background: linear-gradient(90deg, #7138e8, #9850ff);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: #a36aff;
}

[data-testid="stFileUploader"] {
    background: #19233b;
    border: 1px dashed #8064c9;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}

.hero {
    background: linear-gradient(120deg, #191d3c, #29204d);
    border: 1px solid #45406b;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 25px;
}

.metric-card {
    background: #19233b;
    border: 1px solid #35415f;
    padding: 20px;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #aab0c5;
    padding: 25px;
}

hr {
    border-color: #303956;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = joblib.load("models/phishing_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")

    return model, feature_names


try:
    model, feature_names = load_model()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# ==================================================
# SESSION STATE
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "qr_result" not in st.session_state:
    st.session_state.qr_result = None

if "qr_upload_id" not in st.session_state:
    st.session_state.qr_upload_id = None

if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None


# ==================================================
# URL ANALYSIS
# ==================================================

def analyze_url(url):

    url = str(url).strip()

    if not url:
        raise ValueError("Please enter a URL.")

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)

    if not parsed.hostname:
        raise ValueError("Invalid website URL.")

    features = extract_features(url)

    feature_df = pd.DataFrame([features])

    feature_df = feature_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    prediction = int(model.predict(feature_df)[0])

    probabilities = model.predict_proba(feature_df)[0]

    classes = list(model.classes_)

    phishing_index = classes.index(1)

    risk_score = float(probabilities[phishing_index]) * 100

    return {
        "url": url,
        "prediction": prediction,
        "risk_score": risk_score,
        "features": features,
        "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    }


# ==================================================
# QR CODE DECODER - ZXING + OPENCV
# ==================================================

def decode_qr(image_bytes):

    # Convert uploaded bytes into an image
    image_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    # Try multiple image versions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    bordered = cv2.copyMakeBorder(
        enlarged,
        60, 60, 60, 60,
        cv2.BORDER_CONSTANT,
        value=255
    )

    _, threshold = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    images = [
        image,
        gray,
        enlarged,
        bordered,
        threshold
    ]

    # ZXing decoder
    for current_image in images:

        try:
            results = zxingcpp.read_barcodes(current_image)

            for result in results:

                if result.text and result.text.strip():
                    return result.text.strip()

        except Exception:
            pass

    # OpenCV fallback
    detector = cv2.QRCodeDetector()

    for current_image in images:

        try:
            decoded, points, _ = detector.detectAndDecode(
                current_image
            )

            if decoded and decoded.strip():
                return decoded.strip()

        except cv2.error:
            pass

    # Try rotated versions
    for rotation in [
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    ]:

        rotated = cv2.rotate(image, rotation)

        try:
            results = zxingcpp.read_barcodes(rotated)

            for result in results:

                if result.text and result.text.strip():
                    return result.text.strip()

        except Exception:
            pass

        try:
            decoded, points, _ = detector.detectAndDecode(rotated)

            if decoded and decoded.strip():
                return decoded.strip()

        except cv2.error:
            pass

    return None


# ==================================================
# SAVE HISTORY
# ==================================================

def save_history(result):

    st.session_state.history.insert(0, {
        "URL": result["url"],
        "Result": (
            "Potential Phishing"
            if result["prediction"] == 1
            else "Classified as Legitimate"
        ),
        "Risk Score": round(result["risk_score"], 2),
        "Time": result["time"]
    })


# ==================================================
# THREAT INDICATORS
# ==================================================

def get_indicators(features):

    checks = {
        "Long URL": "url_length",
        "Long hostname": "hostname_length",
        "Multiple subdomains": "subdomain_count",
        "Suspicious words": "suspicious_word_count",
        "IP address in URL": "having_ip_address",
        "URL shortener": "shortening_service",
        "Punycode detected": "punycode",
        "Unusual TLD": "unusual_tld",
        "Suspicious extension": "suspicious_extension"
    }

    indicators = []

    for label, key in checks.items():

        if key in features:

            try:
                if float(features[key]) > 0:
                    indicators.append(label)

            except (ValueError, TypeError):
                pass

    return indicators or [
        "No obvious suspicious URL indicators detected."
    ]


# ==================================================
# DISPLAY RESULT
# ==================================================

def display_result(result):

    st.markdown("---")

    st.subheader("Security Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Phishing Risk Score",
            f'{result["risk_score"]:.2f}%'
        )

    with col2:

        if result["prediction"] == 1:
            st.error("⚠️ Potential Phishing")
        else:
            st.success("✅ Classified as Legitimate")

    st.progress(
        max(0.0, min(1.0, result["risk_score"] / 100))
    )

    if result["prediction"] == 1:
        st.warning(
            "The model classified this URL as potentially malicious. "
            "Avoid entering personal information."
        )
    else:
        st.success(
            "The model classified this URL as legitimate. "
            "This does not guarantee website safety."
        )

    st.subheader("URL Threat Indicators")

    for item in get_indicators(result["features"]):
        st.write("•", item)

    with st.expander("View Technical Features"):

        feature_df = pd.DataFrame(
            list(result["features"].items()),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            feature_df,
            use_container_width=True
        )

    report = pd.DataFrame([{
        "URL": result["url"],
        "Prediction": (
            "Phishing" if result["prediction"] == 1
            else "Legitimate"
        ),
        "Risk Score": round(result["risk_score"], 2),
        "Time": result["time"]
    }])

    st.download_button(
        "⬇️ Download Scan Report",
        report.to_csv(index=False),
        "phishguard_report.csv",
        "text/csv"
    )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🛡️ PhishGuard AI")

    st.caption("AI-Powered Phishing Detection")

    st.divider()

    st.subheader("Model Information")

    st.metric("Training URLs", "9,046")

    st.metric("Unseen-Hostname Accuracy", "88.50%")

    st.divider()

    st.subheader("About")

    st.write(
        "PhishGuard AI uses machine learning to identify "
        "potentially malicious website links."
    )

    st.caption("Educational project. Predictions are estimates.")


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>🛡️ PhishGuard AI</h1>
    <h3>Your AI-Powered Phishing Detection Companion</h3>
    <p>
        Detect suspicious website links using machine learning.
        Scan QR codes and analyse multiple URLs to identify
        potential online threats.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 URL Scanner",
    "📷 QR Code Scanner",
    "📁 Bulk Scanner",
    "📊 Scan History"
])


# ==================================================
# TAB 1 - URL SCANNER
# ==================================================

with tab1:

    st.header("Website URL Security Scanner")

    st.write("Enter a website URL to check its potential phishing risk.")

    url_input = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        key="url_input"
    )

    if st.button("🔍 Analyse Website URL"):

        if not url_input.strip():
            st.warning("Please enter a URL.")

        else:

            try:

                with st.spinner("Analysing URL..."):

                    result = analyze_url(url_input)
                    save_history(result)

                display_result(result)

            except Exception as error:
                st.error(f"Analysis failed: {error}")


# ==================================================
# TAB 2 - QR CODE SCANNER
# ==================================================

with tab2:

    st.header("QR Code Security Scanner")

    st.write(
        "Upload a QR code image. PhishGuard AI will decode its "
        "contents and analyse website links."
    )

    uploaded_qr = st.file_uploader(
        "Upload QR Code Image",
        type=["png", "jpg", "jpeg"],
        key="qr_upload"
    )

    # Clear old decoded result when a new image is uploaded
    current_upload_id = (
        uploaded_qr.file_id if uploaded_qr is not None else None
    )

    if current_upload_id != st.session_state.qr_upload_id:

        st.session_state.qr_upload_id = current_upload_id
        st.session_state.qr_result = None

    if uploaded_qr is not None:

        st.image(
            uploaded_qr,
            caption="Uploaded QR Code",
            width=300
        )

        if st.button("📷 Decode QR Code", key="decode_button"):

            with st.spinner("Decoding QR code..."):

                try:
                    decoded_text = decode_qr(
                        uploaded_qr.getvalue()
                    )

                    st.session_state.qr_result = decoded_text

                except Exception as error:

                    st.error(f"QR decoding error: {error}")

        decoded_text = st.session_state.qr_result

        if decoded_text:

            st.success("QR code successfully decoded!")

            st.subheader("Extracted QR Content")

            st.code(decoded_text)

            if decoded_text.lower().startswith(
                ("http://", "https://")
            ):

                st.info("Website link detected.")

                if st.button(
                    "🛡️ Analyse QR Website",
                    key="analyse_qr_button"
                ):

                    try:

                        with st.spinner("Analysing website..."):

                            result = analyze_url(decoded_text)
                            save_history(result)

                        display_result(result)

                    except Exception as error:
                        st.error(f"Analysis failed: {error}")

            else:

                st.warning(
                    "This QR code does not contain a direct HTTP or "
                    "HTTPS link. Review the decoded content carefully."
                )

        elif decoded_text is None:

            st.info(
                "Click 'Decode QR Code' to read the uploaded image."
            )

    else:

        st.info("Upload a QR code image to begin.")


# ==================================================
# TAB 3 - BULK SCANNER
# ==================================================

with tab3:

    st.header("Bulk URL Security Scanner")

    st.write(
        "Upload a CSV file containing a column named 'url'."
    )

    uploaded_csv = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key="csv_upload"
    )

    if uploaded_csv is not None:

        try:

            data = pd.read_csv(uploaded_csv)

            if "url" not in data.columns:

                st.error("Your CSV must contain a 'url' column.")

            elif len(data) == 0:

                st.warning("The CSV file is empty.")

            else:

                st.dataframe(
                    data.head(10),
                    use_container_width=True
                )

                if st.button("🚀 Scan All URLs"):

                    results = []

                    progress = st.progress(0)

                    for index, row in data.iterrows():

                        try:

                            result = analyze_url(row["url"])

                            save_history(result)

                            results.append({
                                "URL": result["url"],
                                "Prediction": (
                                    "Phishing"
                                    if result["prediction"] == 1
                                    else "Legitimate"
                                ),
                                "Risk Score": round(
                                    result["risk_score"], 2
                                )
                            })

                        except Exception as error:

                            results.append({
                                "URL": str(row["url"]),
                                "Prediction": "Error",
                                "Risk Score": str(error)
                            })

                        progress.progress(
                            (index + 1) / len(data)
                        )

                    st.session_state.bulk_results = pd.DataFrame(
                        results
                    )

                if st.session_state.bulk_results is not None:

                    st.subheader("Bulk Scan Results")

                    st.dataframe(
                        st.session_state.bulk_results,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Download Bulk Report",
                        st.session_state.bulk_results.to_csv(
                            index=False
                        ),
                        "phishguard_bulk_report.csv",
                        "text/csv"
                    )

        except Exception as error:

            st.error(f"CSV processing failed: {error}")


# ==================================================
# TAB 4 - SCAN HISTORY
# ==================================================

with tab4:

    st.header("Scan History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Scan History",
            history_df.to_csv(index=False),
            "phishguard_history.csv",
            "text/csv"
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []
            st.rerun()

    else:

        st.info("No scan history available yet.")


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <h3>🛡️ PhishGuard AI</h3>
    AI-Based Phishing URL Detection System
    <br><br>
    Built using Python, Machine Learning, OpenCV and Streamlit.
    <br>
    Educational project. Predictions do not guarantee website safety.
</div>
""", unsafe_allow_html=True)
