# database.py
import mysql.connector

# ================= DATABASE CONFIG =================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "mediscan_ai"
}

# ================= GET CONNECTION =================
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

# ================= SAVE PREDICTION =================
def save_prediction(
    patient_name,
    age,
    gender,
    scan_type,
    prediction,
    confidence,
    risk_level,
    original_image,
    heatmap_image
):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO predictions
        (patient_name, age, gender, scan_type,
         prediction, confidence, risk_level,
         original_image, heatmap_image)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        values = (
            patient_name,
            age,
            gender,
            scan_type,
            prediction,
            confidence,
            risk_level,
            original_image,
            heatmap_image
        )

        cursor.execute(query, values)
        conn.commit()

        cursor.close()
        conn.close()

        print("✅ Prediction saved to database")

    except Exception as e:
        print("❌ Database Error:", e)

