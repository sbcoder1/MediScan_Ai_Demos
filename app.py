from flask import Flask, render_template, request, url_for
import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

from flask import Flask, render_template, request, url_for, send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import csv
import io
from datetime import datetime

from database import save_prediction, get_connection



# ================= CREATE APP =================
app = Flask(__name__, template_folder="templates")



# ================= GLOBAL STORAGE =================
latest_original_image = None
latest_heatmap_image = None
latest_patient_name = None
latest_confidence = None
latest_risk_level = None
latest_age = None
latest_gender = None
latest_prediction = None
latest_scan_date = None
latest_patient_id = None
latest_contact = None
latest_doctor = None
latest_scan_type = None
latest_module = None   # "brain" | "lung" | "liver"
latest_heart_score = None
latest_heart_risk = None



# ================= FOLDERS =================
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ================= IMAGE SIZES =================
BRAIN_CNN_SIZE = (64, 64)
BRAIN_UNET_SIZE = (256, 256)
LUNG_IMG_SIZE = (224, 224)

# ================= LOAD MODELS =================
brain_cnn = load_model("models/BrainTumor10Epochs.h5")
brain_unet = load_model("models/unet_brain_mri.h5")
lung_model = load_model("models/lung_disease_model.h5")

# ================= LOAD LIVER MODEL =================
liver_model = load_model("models/liver_disease_model.h5")

# IMPORTANT: order must match training
LIVER_CLASSES = ["Liver_Tumor", "Normal"]


# Warm-up
brain_cnn.predict(np.zeros((1, 64, 64, 3), dtype="float32"))
brain_unet.predict(np.zeros((1, 256, 256, 3), dtype="float32"))
lung_model.predict(np.zeros((1, 224, 224, 3), dtype="float32"))
liver_model.predict(np.zeros((1,128,128,3), dtype="float32"))


LUNG_CLASSES = ["Normal", "COVID-19", "Pneumonia", "Lung Opacity"]

# ================= HOME ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/brain")
def brain_form():
    return render_template("brain.html")

@app.route("/lungs")
def lungs_form():
    return render_template("lungs.html")

@app.route("/liver")
def liver_form():
    return render_template("liver.html")

@app.route("/heart")
def heart_form():
    return render_template("heart.html")

@app.route("/logout")
def logout():
    session.clear()
    return jsonify({"success": True})

# ================= PDF DOWNLOAD =================
@app.route("/download_pdf")
def download_pdf():
    global latest_original_image, latest_heatmap_image
    global latest_patient_name, latest_confidence, latest_risk_level
    global latest_age, latest_gender, latest_patient_id
    global latest_contact, latest_doctor, latest_scan_type
    global latest_module, latest_prediction

    patient_name = latest_patient_name or "Unknown"
    age = latest_age or "N/A"
    gender = latest_gender or "N/A"

    safe_name = patient_name.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"MediScan_Report_{safe_name}_{timestamp}.pdf"

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ================= HEADER =================
    c.setFillColorRGB(0.02, 0.25, 0.18)
    c.rect(0, height - 80, width, 80, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 50, "MediScan AI")
    c.setFont("Helvetica", 12)

    # -------- Dynamic Report Title --------
    if latest_module == "lung":
        report_title = "AI-Based Lung Disease Diagnostic Report"
    elif latest_module == "liver":
        report_title = "AI-Based Liver Disease Diagnostic Report"
    elif latest_module == "heart":
        report_title = "AI-Based Heart Disease Risk Assessment Report"
    else:
        report_title = "AI-Based Brain Tumor Diagnostic Report"

    c.drawString(50, height - 70, report_title)
    c.setFillColorRGB(0, 0, 0)

    # ================= PATIENT INFO =================
    y = height - 120
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Patient Information")

    c.setFont("Helvetica", 11)
    y -= 22

    # Patient ID (hide N/A for heart)
    if latest_module != "heart":
        c.drawString(50, y, f"Patient ID       : {latest_patient_id or 'N/A'}")
    elif latest_patient_id:
        c.drawString(50, y, f"Patient ID       : {latest_patient_id}")

    c.drawString(300, y, f"Age : {age}")
    y -= 18

    # Name & Gender (always shown)
    c.drawString(50, y, f"Patient Name     : {patient_name}")
    c.drawString(300, y, f"Gender : {gender}")
    y -= 18

    # Contact Number
    if latest_module != "heart":
        c.drawString(50, y, f"Contact Number   : {latest_contact or 'N/A'}")
    elif latest_contact:
        c.drawString(50, y, f"Contact Number   : {latest_contact}")

    # Scan Type
    if latest_module != "heart":
        c.drawString(300, y, f"Scan Type : {latest_scan_type or 'N/A'}")
    elif latest_scan_type:
        c.drawString(300, y, f"Scan Type : {latest_scan_type}")

    y -= 18

    # Referring Doctor
    if latest_module != "heart":
        c.drawString(50, y, f"Referring Doctor : {latest_doctor or 'N/A'}")
    elif latest_doctor:
        c.drawString(50, y, f"Referring Doctor : {latest_doctor}")

    y -= 18

    # Date & Time (always shown)
    c.drawString(
        50, y,
        f"Date & Time      : {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
    )

    # ================= DIAGNOSIS =================
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Diagnosis Summary")

    c.setFont("Helvetica", 11)
    y -= 18

    if latest_module == "heart":
        c.drawString(50, y, f"Risk Assessment  : {latest_prediction or 'N/A'}")
    elif latest_module in ["lung", "liver"]:
        c.drawString(50, y, f"Diagnosis        : {latest_prediction or 'N/A'}")
    else:
        c.drawString(50, y, "Diagnosis        : Brain Tumor Detected")

    y -= 16
    c.drawString(50, y, f"Confidence Score : {latest_confidence or 'N/A'}%")
    y -= 16
    c.drawString(50, y, f"Risk Level       : {latest_risk_level or 'N/A'}")

    # ================= IMAGES =================
    if latest_module != "heart":
        y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Imaging Results")

        if latest_original_image and os.path.exists(latest_original_image):
            c.drawImage(latest_original_image, 50, y - 230, width=220, height=200)

        if latest_heatmap_image and os.path.exists(latest_heatmap_image):
            c.drawImage(latest_heatmap_image, 320, y - 230, width=220, height=200)

    # ================= FOOTER =================
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        50, 40,
        "Disclaimer: AI-assisted result. Must be reviewed by a qualified medical professional."
    )

    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf"
    )

@app.route("/export_data")
def export_data():
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Module",
        "Patient Name",
        "Age",
        "Gender",
        "Scan Type",
        "Scan Date",
        "Prediction",
        "Confidence (%)",
        "Risk Level"
    ])

    # Data row
    writer.writerow([
        latest_module,
        latest_patient_name,
        latest_age,
        latest_gender,
        latest_scan_type,
        latest_scan_date,
        latest_prediction,
        latest_confidence,
        latest_risk_level
    ])

    output.seek(0)

    filename = f"mediscan_{latest_module}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )
    
# ================= BRAIN PREDICTION (FIXED) =================
@app.route("/brain/predict", methods=["POST"])
def brain_predict():
    global latest_original_image, latest_heatmap_image
    global latest_confidence, latest_risk_level
    global latest_patient_name, latest_age, latest_gender, latest_scan_date
    global latest_module, latest_scan_type
    global latest_patient_id, latest_contact, latest_doctor

    latest_module = "brain"
    latest_scan_type = "Brain MRI"

    # ===== PATIENT DATA =====
    latest_patient_name = request.form.get("patient_name")
    latest_age = request.form.get("age")
    latest_gender = request.form.get("gender")
    latest_scan_date = request.form.get("scan_date")
    latest_patient_id = request.form.get("patient_id")
    latest_contact = request.form.get("contact")
    latest_doctor = request.form.get("doctor_name")
    latest_scan_type = request.form.get("scan_type")

    # ===== FILE =====
    file = request.files["scan_file"]
    img_name = f"brain_{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_FOLDER, img_name)
    file.save(path)

    # ===== PREDICT =====
    img = preprocess_brain_cnn(path)
    prob = float(brain_cnn.predict(img)[0][0])

    latest_confidence = round(prob * 100, 2)
    latest_risk_level = brain_risk_level(prob)
    latest_original_image = path
    latest_prediction = "Brain Tumor Detected" if prob >= 0.5 else "No Brain Tumor"

    # ===== HEATMAP =====
    latest_heatmap_image = None
    overlay_url = None

    if prob >= 0.60:
        mask = segment_brain(path)
        overlay_name = overlay_mask(path, mask)
        latest_heatmap_image = os.path.join(RESULT_FOLDER, overlay_name)
        overlay_url = url_for("static", filename=f"results/{overlay_name}")

    # ===== SAVE TO DATABASE =====
    save_prediction(
        latest_patient_name,
        latest_age,
        latest_gender,
        latest_scan_type,
        latest_prediction,
        latest_confidence,
        latest_risk_level,
        latest_original_image,
        latest_heatmap_image
    )

    return render_template(
        "brain_result.html",
        patient_name=latest_patient_name,
        age=latest_age,
        gender=latest_gender,
        scan_date=latest_scan_date,
        predicted_label=latest_prediction,
        prob_percent=latest_confidence,
        risk_level=latest_risk_level,
        original_url=url_for("static", filename=f"uploads/{img_name}"),
        heatmap_url=overlay_url
    )

# ================= LUNG PREDICTION (FIXED) =================
@app.route("/lung/predict", methods=["POST"])
def lung_predict():
    global latest_original_image, latest_heatmap_image
    global latest_confidence, latest_risk_level, latest_prediction
    global latest_patient_name, latest_age, latest_gender, latest_scan_date
    global latest_module, latest_scan_type
    global latest_patient_id, latest_contact, latest_doctor, latest_scan_type
    
    latest_module = "lung"
    latest_scan_type = "Chest X-ray"

    # ===== PATIENT DATA (FIXED) =====
    latest_patient_name = request.form.get("patient_name")
    latest_age = request.form.get("age")
    latest_gender = request.form.get("gender")
    latest_scan_date = request.form.get("scan_date")
    latest_patient_id = request.form.get("patient_id")
    latest_contact = request.form.get("contact")
    latest_doctor = request.form.get("doctor_name")
    # scan type (Chest X-ray)
    latest_scan_type = request.form.get("scan_type")

    # ===== FILE =====
    file = request.files["scan_file"]
    img_name = f"lung_{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_FOLDER, img_name)
    file.save(path)

    # ===== PREPROCESS =====
    img_array = preprocess_lung(path)

    # ===== PREDICT =====
    preds = lung_model.predict(img_array)[0]
    preds = [float(p) for p in preds]

    class_id = int(np.argmax(preds))
    confidence = preds[class_id]

    latest_prediction = LUNG_CLASSES[class_id]
    latest_confidence = round(confidence * 100, 2)
    latest_original_image = path

    # ===== RISK =====
    if latest_prediction == "Normal":
        latest_risk_level = "Low Risk"
    elif latest_prediction == "COVID-19":
        latest_risk_level = "High Risk"
    else:
        latest_risk_level = "Moderate Risk"

    # ===== GRAD-CAM =====
    heatmap = generate_gradcam(img_array, lung_model)
    overlay_name = overlay_gradcam(path, heatmap)
    latest_heatmap_image = os.path.join(RESULT_FOLDER, overlay_name)

    # ===== SAVE TO DATABASE ===== ✅ ADDED
    save_prediction(
        latest_patient_name,
        latest_age,
        latest_gender,
        latest_scan_type,
        latest_prediction,
        latest_confidence,
        latest_risk_level,
        latest_original_image,
        latest_heatmap_image
    )
    
    return render_template(
        "lungs_result.html",
        patient_name=latest_patient_name,
        age=latest_age,
        gender=latest_gender,
        scan_date=latest_scan_date,
        predicted_label=latest_prediction,
        confidence=latest_confidence,
        risk_level=latest_risk_level,
        original_url=url_for("static", filename=f"uploads/{img_name}"),
        heatmap_url=url_for("static", filename=f"results/{overlay_name}"),
        class_probs={
            "Normal": round(preds[0] * 100, 2),
            "COVID-19": round(preds[1] * 100, 2),
            "Pneumonia": round(preds[2] * 100, 2),
            "Lung Opacity": round(preds[3] * 100, 2),
        }
    )

# ================= HEART PREDICTION =================
@app.route("/heart/predict", methods=["POST"])
def heart_predict():
    global latest_patient_name, latest_age, latest_gender
    global latest_contact, latest_confidence, latest_risk_level
    global latest_module, latest_heart_score, latest_prediction

    latest_module = "heart"

    # ================= PATIENT DATA =================
    latest_patient_name = request.form.get("patient_name")
    latest_age = int(request.form.get("age"))
    latest_gender = request.form.get("gender")
    latest_contact = request.form.get("contact")

    # ================= CLINICAL DATA =================
    bp = float(request.form.get("bp"))
    cholesterol = float(request.form.get("cholesterol"))
    blood_sugar = float(request.form.get("blood_sugar") or 0)
    heart_rate = float(request.form.get("heart_rate") or 0)
    chest_pain = request.form.get("chest_pain")

    # ================= RISK SCORING (ML-STYLE) =================
    risk_score = 0

    if bp > 140: risk_score += 20
    if cholesterol > 240: risk_score += 25
    if blood_sugar > 120: risk_score += 15
    if heart_rate < 100: risk_score += 10

    if chest_pain in ["Typical Angina", "Asymptomatic"]:
        risk_score += 30
    elif chest_pain == "Atypical Angina":
        risk_score += 20
    else:
        risk_score += 10

    risk_score = min(risk_score, 100)
    latest_heart_score = risk_score

    # ================= CLASSIFICATION =================
    if risk_score < 40:
        latest_prediction = "Low Risk of Heart Disease"
        latest_risk_level = "Low Risk"
    elif risk_score < 70:
        latest_prediction = "Moderate Risk of Heart Disease"
        latest_risk_level = "Moderate Risk"
    else:
        latest_prediction = "High Risk of Heart Disease"
        latest_risk_level = "High Risk"

    latest_confidence = risk_score  # treated as confidence %

    # ================= SAVE TO DATABASE =================
    save_prediction(
        latest_patient_name,
        latest_age,
        latest_gender,
        latest_scan_type,
        latest_prediction,
        latest_confidence,
        latest_risk_level,
        None,   # ❗ No original image
        None    # ❗ No heatmap
    )
    
    return render_template(
        "heart_result.html",
        patient_name=latest_patient_name,
        age=latest_age,
        gender=latest_gender,
        predicted_label=latest_prediction,
        confidence=latest_confidence,
        risk_level=latest_risk_level,
        bp=bp,
        cholesterol=cholesterol,
        blood_sugar=blood_sugar,
        heart_rate=heart_rate,
        chest_pain=chest_pain
    )

# ================= LIVER PREDICTION =================
@app.route("/liver/predict", methods=["POST"])
def liver_predict():
    global latest_original_image, latest_heatmap_image
    global latest_confidence, latest_risk_level, latest_prediction
    global latest_patient_name, latest_age, latest_gender, latest_scan_date
    global latest_patient_id, latest_contact, latest_doctor, latest_scan_type
    global latest_module

    latest_module = "liver"

    # ================= PATIENT DATA =================
    latest_patient_name = request.form.get("patient_name")
    latest_age = request.form.get("age")
    latest_gender = request.form.get("gender")
    latest_scan_date = request.form.get("scan_date")

    latest_patient_id = request.form.get("patient_id")
    latest_contact = request.form.get("contact")
    latest_doctor = request.form.get("doctor_name")
    latest_scan_type = request.form.get("scan_type")

    # ================= FILE =================
    file = request.files["scan_file"]
    img_name = f"liver_{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_FOLDER, img_name)
    file.save(path)

    latest_original_image = path

    # ================= PREPROCESS =================
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ================= PREDICT =================
    preds = liver_model.predict(img)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    latest_prediction = LIVER_CLASSES[class_id]
    latest_confidence = round(confidence * 100, 2)

    # ================= RISK LEVEL =================
    latest_risk_level = "Low Risk" if latest_prediction == "Normal" else "High Risk"

    # ================= LIVER ATTENTION MAP (SAFE) =================
    heatmap = generate_liver_attention(path)
    overlay_name = overlay_liver_attention(path, heatmap)
    latest_heatmap_image = os.path.join(RESULT_FOLDER, overlay_name)

    save_prediction(
        latest_patient_name,
        latest_age,
        latest_gender,
        latest_scan_type,
        latest_prediction,
        latest_confidence,
        latest_risk_level,
        latest_original_image,
        latest_heatmap_image
    )
    
    return render_template(
    "liver_result.html",
    patient_name=latest_patient_name,
    age=latest_age,
    gender=latest_gender,
    scan_date=latest_scan_date,
    predicted_label=latest_prediction,
    confidence=latest_confidence,
    risk_level=latest_risk_level,
    original_url=url_for("static", filename=f"uploads/{img_name}"),
    heatmap_url=url_for("static", filename=f"results/{overlay_name}"),
)

    
# ================= HELPERS =================
def preprocess_brain_cnn(path):
    img = cv2.resize(cv2.imread(path), BRAIN_CNN_SIZE) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_lung(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, LUNG_IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def brain_risk_level(p):
    if p < 0.40: return "Likely No Tumor"
    elif p < 0.60: return "Uncertain"
    elif p < 0.80: return "Tumor Likely"
    return "Strong Tumor Indication"

def segment_brain(path):
    img = cv2.resize(cv2.imread(path), BRAIN_UNET_SIZE) / 255.0
    img = np.expand_dims(img, axis=0)
    return (brain_unet.predict(img)[0, :, :, 0] > 0.5).astype("uint8")

def overlay_mask(path, mask):
    img = cv2.imread(path)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img[mask == 1] = [0, 0, 255]
    name = f"brain_overlay_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join(RESULT_FOLDER, name), img)
    return name

# ================= GRAD-CAM (SAFE FOR ALL MODELS) =================
def generate_gradcam(img_array, model):
    # 🔹 Force model build (VERY IMPORTANT)
    _ = model(img_array)

    # 🔹 Automatically find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in the model for Grad-CAM")

    # 🔹 Build Grad-CAM model safely
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


#Lungs 
def overlay_gradcam(path, heatmap):
    img = cv2.imread(path)
    img = cv2.resize(img, LUNG_IMG_SIZE)

    heatmap = cv2.resize(heatmap, LUNG_IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    name = f"lung_cam_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join(RESULT_FOLDER, name), overlay)
    return name

# ================= LIVER ATTENTION (SAFE REPLACEMENT) =================
def generate_liver_attention(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = gray.astype("float32") / 255.0
    heatmap = cv2.GaussianBlur(gray, (21, 21), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)

    return heatmap


def overlay_liver_attention(path, heatmap):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)

    name = f"liver_attention_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join(RESULT_FOLDER, name), overlay)
    return name


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
