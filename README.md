# CT Bowel Injury Detection

ระบบประเมินความเสี่ยงการบาดเจ็บลำไส้จากภาพเอกซเรย์คอมพิวเตอร์ช่องท้อง โดยใช้ปัญญาประดิษฐ์
(AI-Based Risk Assessment of Bowel Injury from Abdominal CT)

> ⚠️ ต้นแบบเพื่อการศึกษาเท่านั้น — ไม่ใช้วินิจฉัยทางคลินิก

---

## Features

- อัปโหลดไฟล์ CT ในรูปแบบ `.npy` หรือ ZIP ที่มี DICOM slices
- ประเมินความเสี่ยงการบาดเจ็บลำไส้เป็นค่าความน่าจะเป็น 0–1
- แสดงระดับความเสี่ยง 3 ระดับ: 🟢 LOW / 🟠 MEDIUM / 🔴 HIGH
- แสดง Saliency Map บริเวณที่โมเดลให้ความสนใจ
- ดาวน์โหลดรายงานผลการประเมินเป็นไฟล์ TXT
- รองรับ Windows และ Linux / GitHub Codespaces

## Model

- **Architecture:** ResNet18 (feature extraction) + GRU (sequence modeling)
- **Input:** CT volume → 2.5D sequence 32 steps × 3 slices/step
- **Dataset:** RSNA 2023 Abdominal Trauma Detection (4,000+ cases)
- **Performance (threshold = 0.7):**

| Metric | Value |
|--------|-------|
| Accuracy | 96.09% |
| Precision | 78.57% |
| Recall | 84.62% |
| F1-score | 81.48% |
| ROC-AUC | 93.78% |

## Run locally

```
pip install -r requirements.txt
streamlit run last.py
```

Open http://localhost:8501

## Run on GitHub Codespaces

1. กด **Code → Codespaces → Create codespace**
2. รอ environment ติดตั้งอัตโนมัติ
3. แอปจะเปิดที่ port 8501

## Training (optional)

```
pip install -r requirements-training.txt
python firstbowel_injury_model.py --data-dir /path/to/data --epochs 50
```

โฟลเดอร์ข้อมูลต้องมี `.npy` volumes และ `labels.csv` (คอลัมน์: `patient_id`, `bowel_injury`)

## File Structure

```
last.py                   # Streamlit web app (inference)
firstbowel_injury_model.py # Training script
requirements.txt           # Inference dependencies
requirements-training.txt  # Training dependencies
```

## Team

- ชากีรีน อาแซ
- อับดุลก็อฟฟาร์ นุ้ยดำ

**อาจารย์ที่ปรึกษา:** อารฝัน บากา
**โรงเรียน:** วิทยาศาสตร์จุฬาภรณราชวิทยาลัย สตูล
