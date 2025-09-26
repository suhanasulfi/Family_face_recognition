

# 👨‍👩‍👧 Family Face Recognition System

A real-time **face recognition system** that identifies family members through a webcam and displays their **name** and **relation** (e.g., “Nisa (Mother of Suhana)”). The project uses **PyTorch** for deep learning and **OpenCV** for face detection.

---

## 🚀 Features

* Real-time **face detection** using OpenCV Haar cascades
* Face classification with **ResNet-18** (transfer learning)
* Displays **name + relation** of detected family members
* Custom dataset of family images
* Supports **precision, recall, accuracy** evaluation

---

## 📂 Project Structure

```
FACE_RECOGNITION_OF_FAMILY_MEMBERS/
│
├── dataset/
│   ├── train/
│   │   ├── fariyal/
│   │   ├── nisa/
│   │   ├── sulfi/
│   │   ├── sulthana/
│   │   └── yaseen/
│   └── val/
│       ├── fariyal/
│       ├── nisa/
│       ├── sulfi/
│       ├── sulthana/
│       └── yaseen/
│
├── train_model.py        # Training script
├── realtime_face.py      # Real-time recognition script
└── face_model.pth        # Saved trained model
```

---

## ⚙️ Installation

1. Clone this repo:

```bash
git clone https://github.com/your-username/family-face-recognition.git
cd family-face-recognition
```

2. Create virtual environment and install dependencies:

```bash
python -m venv venv
# Activate venv
venv\Scripts\activate   # Windows  
source venv/bin/activate  # Mac/Linux

# Install packages
pip install torch torchvision opencv-python pillow
```

3. Prepare dataset:

   * Place images in `dataset/train` and `dataset/val` with **one folder per person**.
   * Example: `dataset/train/nisa/`, `dataset/val/nisa/`.

---

## 🏋️ Training the Model

Run the training script:

```bash
python train_model.py
```

* Trains **ResNet-18** on your dataset
* Saves model as `face_model.pth`

---

## 🎥 Real-Time Recognition

Run the recognition script:

```bash
python realtime_face.py
```

* Opens webcam
* Detects faces and predicts **name + relation**
* Press **q** to quit

---

## 📊 Results

Evaluation on validation set:

* **Precision** ≈ *your value*
* **Recall** ≈ *your value*
* **Accuracy** ≈ *your value*

(Screenshot examples here)

---

## 🛠️ Technologies Used

* **PyTorch** – Deep learning framework
* **torchvision** – Pretrained ResNet-18 model + transforms
* **OpenCV** – Haar cascade for face detection, display
* **Pillow (PIL)** – Image handling

---

## 📖 Future Improvements

* Add more family members with bigger dataset
* Use **MTCNN** or **dlib** for better face detection
* Data augmentation for improved accuracy
* Deploy as a **desktop or mobile app**

---

## 👩‍💻 Author

**Suhana P S**
B.Tech Artificial Intelligence & Data Science

---


