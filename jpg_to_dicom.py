import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from PIL import Image
import numpy as np
import datetime
import os

def jpg_to_dicom(jpg_path, dicom_path):
    img = Image.open(jpg_path).convert("L")
    pixel_array = np.array(img)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        dicom_path,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )

    ds.Modality = "CT"
    ds.ContentDate = datetime.date.today().strftime("%Y%m%d")
    ds.ContentTime = datetime.datetime.now().strftime("%H%M%S")
    ds.PatientName = "CT^Brain"
    ds.PatientID = "0001"

    ds.Rows, ds.Columns = pixel_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7

    ds.PixelData = pixel_array.tobytes()
    ds.save_as(dicom_path)

    print(f"✅ Converted: {os.path.basename(dicom_path)}")


def convert_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            jpg_path = os.path.join(input_folder, file)
            dcm_name = os.path.splitext(file)[0] + ".dcm"
            dicom_path = os.path.join(output_folder, dcm_name)
            jpg_to_dicom(jpg_path, dicom_path)


# ================= MAIN =================
if __name__ == "__main__":

    base_input = r"D:\MediScan_Ai\Images_CT_Brain"
    base_output = r"D:\MediScan_Ai\DICOM_CT_Brain"

    classes = ["tumor", "yes"]

    for cls in classes:
        convert_folder(
            os.path.join(base_input, cls),
            os.path.join(base_output, cls)
        )

    print("🎉 ALL IMAGES CONVERTED TO DICOM SUCCESSFULLY")

