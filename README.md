# GAN Learning Visualization

This project demonstrates how a Generative Adversarial Network (GAN) learns over time by visualizing outputs at different epochs.

---

## 📌 Objective
To understand how GANs improve gradually by observing generated outputs across training epochs.

---

## 🧠 What is GAN?
A GAN (Generative Adversarial Network) consists of two models:

- **Generator** → Generates fake data  
- **Discriminator** → Distinguishes real vs fake  

Both models compete and improve together.

---

## ⚙️ Implementation
- Built using **PyTorch**
- Trained on a **2D heart-shaped dataset**
- Outputs captured at different epochs

---

## 📊 Training Progress

### 🔹 Original Training Data
<img width="599" height="605" alt="image" src="https://github.com/user-attachments/assets/ba5907dd-a4af-49a1-9150-b99e8516d69e" />



---

### 🔹 Epoch 500
Initial learning stage — model starts forming patterns
<img width="613" height="608" alt="image" src="https://github.com/user-attachments/assets/9922e912-2d9d-432a-98fc-e987b43c5d66" />



---

### 🔹 Epoch 2500
Gradual improvement — structure starts becoming visible

<img width="624" height="618" alt="image" src="https://github.com/user-attachments/assets/82b57a8d-7705-48c8-8699-65cc161ad646" />



---

### 🔹 Final Epoch (4000)
Model has learned the general pattern, but still improving
<img width="602" height="617" alt="image" src="https://github.com/user-attachments/assets/1eeed5f2-eafb-4759-a1c8-7f5c2f5f0e49" />


---

## 📈 Observations
- Learning is **gradual**, not immediate  
- Early outputs are **random noise**  
- Model slowly captures the **data distribution**  
- Results are not perfect but show **clear improvement**

---

## 📚 Learnings
- GAN training requires patience  
- Visualization helps in understanding model behavior  
- Practical implementation improves conceptual clarity  

---

## 🚀 Future Improvements
- Train for more epochs  
- Tune hyperparameters  
- Use more complex datasets  

---

## 🔗 Author
This is my **first GAN experiment** as part of learning Deep Learning.
