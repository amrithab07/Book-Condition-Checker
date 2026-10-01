# 📚 Book Condition Checker

A web app that classifies a book's condition as **Good** or **Damaged** from a photo, using a MobileNetV2 transfer-learning model served through Flask. Each analysis is stored in MongoDB with a confidence score so a book's condition can be tracked.

---

## 📌 Overview

Checking the condition of used or library books by eye is slow and inconsistent. This project automates a first-pass grading: upload a photo of a book, and the app returns a condition label and confidence score in seconds.

- **Model:** MobileNetV2 (ImageNet pre-trained) with a custom classification head
- **Task:** Binary image classification (Good vs. Damaged)
- **Backend:** Flask REST endpoint
- **Storage:** MongoDB (analysis history per book ID)
- **Dataset:** Self-collected, 325 book images

---

## 🗂️ Dataset

The dataset was collected and labelled by me.

| Class | Images |
|---|---|
| Damaged books | 148 |
| Good books | 177 |
| **Total** | **325** |

- Folder structure used for training: `damaged_books/` and `good_books/`
- Split: 68% train / 12% validation / 20% test
- Images are resized to 224 × 224 and scaled to [0, 1]

<!-- Add a line here describing what counts as "Damaged" (e.g. torn pages, cover wear, water damage, spine damage) and how you took the photos. -->

---

## 🧠 Model

Defined in `train_model.py`.

| Component | Detail |
|---|---|
| Base model | MobileNetV2, ImageNet weights, frozen |
| Head | GlobalAveragePooling → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3) → Dense(1, sigmoid) |
| Loss / optimizer | Binary cross-entropy, Adam (lr = 0.001) |
| Augmentation | Rotation (20°), width/height shift (20%), horizontal flip |
| Training | Batch size 32, up to 10 epochs |
| Callbacks | `ModelCheckpoint` (best `val_accuracy`), `EarlyStopping` (patience 5 on `val_loss`) |

**Results**

| Metric | Value |
|---|---|
| Validation accuracy | 78.5% |
| Damaged class: precision / recall | 0.71 / 0.86 |
| Good class: precision / recall | 0.87 / 0.72 |
| Decision threshold | 0.65 |

<!-- Add a confusion matrix image here, e.g. ![Confusion matrix](images/confusion_matrix.png) -->

---

## 🌐 Web App

Defined in `app.py`.

**Flow**
1. User uploads a `.png`, `.jpg` or `.jpeg` image (max 16 MB) and optionally a `book_id`.
2. The image is converted to RGB, resized to 224 × 224 and normalised.
3. The model outputs a probability. Probability ≥ 0.5 → **Good**, otherwise **Damaged**. Confidence is the probability of the predicted class.
4. The result is saved to MongoDB and returned as JSON.

**Re-analysis of the same book:** if a `book_id` already has a stored result, the app compares confidence values and keeps the lower-confidence entry (the more conservative reading), returning an alert message explaining which result was kept.

**API**

`POST /upload` (multipart form-data)

| Field | Description |
|---|---|
| `file` | Book image (png / jpg / jpeg) |
| `book_id` | Optional book identifier |

Example response:
```json
{
  "condition": "Damaged",
  "confidence": 0.91,
  "alert": null
}
```

**Stored in MongoDB** (`book_condition_db.analysis_results`): `book_id`, `filename`, `condition`, `confidence`, `timestamp`.

---

## 📁 Project Structure

```
├── app.py                    # Flask app: upload, prediction, MongoDB storage
├── start_server.py           # Server entry point (HOST, PORT, DEBUG via env vars)
├── train_model.py            # Model training script
├── book_condition_model.h5   # Final trained model (loaded by the app)
├── best_model.h5             # Best checkpoint by validation accuracy
├── templates/
│   └── index.html            # Drag-and-drop upload UI with live preview
├── uploads/                  # Uploaded images / training data
└── README.md
```

---

## 🛠️ Tech Stack

- **ML:** TensorFlow / Keras, MobileNetV2, NumPy, Pillow
- **Backend:** Flask, Werkzeug
- **Database:** MongoDB (PyMongo)
- **Frontend:** HTML, CSS, JavaScript (drag-and-drop upload with live preview)

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/Book-Condition-Checker.git
   cd Book-Condition-Checker
   ```
2. **Install dependencies**
   ```bash
   pip install tensorflow flask pillow numpy pymongo
   ```
3. **Start MongoDB** locally on `mongodb://localhost:27017/`.
4. **(Optional) Retrain the model.** Arrange images as `uploads/damaged_books/` and `uploads/good_books/`, then:
   ```bash
   python train_model.py
   ```
5. **Run the app**
   ```bash
   python start_server.py
   ```
   Then open `http://localhost:5000`.

Optional environment variables: `HOST` (default `0.0.0.0`), `PORT` (default `5000`), `DEBUG` (default `false`).

---

## ⚠️ Limitations

- **Small dataset.** 325 images means the validation set is only about 65 images, so accuracy figures are noisy and may not generalise to different lighting, backgrounds or book types.
- **Binary labels only.** The model separates Good from Damaged and doesn't grade severity or damage type.
- **No separate test set.** Reported accuracy comes from the validation split, not an unseen test set.

## 🔮 Future Improvements

- Collect more images and hold out a separate test set
- Fine-tune the top layers of MobileNetV2
- Multi-class grading (e.g. Like New / Good / Fair / Poor) and damage-type detection
- Grad-CAM heatmaps to show which part of the book drove the prediction
- Containerise with Docker and deploy

---

## 👩‍💻 Author

**Baratam Amritha**
B.Tech Computer Science Engineering, VIT Chennai

- GitHub: [@amrithab07](https://github.com/amrithab07)
- LinkedIn: [amritha-baratam](https://www.linkedin.com/in/amritha-baratam)
