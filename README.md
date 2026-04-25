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

## 📊 Data Generation
```py
#training data
TRAIN_DATA_COUNT = 1024
theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)])
# Heart shape dataset
x = 16 * ( np.sin(theta) ** 3 )
y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
sns.scatterplot(x=x, y=y)
```
---

## 🧠 GAN Architecture

```py
# Discriminator
discriminator=nn.Sequential(
    nn.Linear(2, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64,1),
    nn.Sigmoid()
)
# Generator
generator=nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16,64),
    nn.ReLU(),
    nn.Linear(64,2)
)
```

## 📊 Training Process
```py
for epoch in range(NUM_EPOCHS):
    for n, (real_samples, _) in enumerate(train_loader):

        # Data for training the discriminator
        real_samples_labels = torch.ones((BATCH_SIZE, 1))
        latent_space_samples = torch.randn((BATCH_SIZE, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((BATCH_SIZE, 1))

        all_samples = torch.cat((real_samples, generated_samples), dim=0)
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels), dim=0)

        # Train discriminator
        if epoch % 2 == 0:
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

        # Train generator
        if epoch % 2 == 1:
            generator.zero_grad()
            latent_space_samples = torch.randn((BATCH_SIZE, 2))
            generated_samples = generator(latent_space_samples)
            output_discriminator_generator = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generator, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

```


## ⚙️ Training Setup

- **Loss Function:** Binary Cross Entropy Loss (BCELoss)  
- **Optimizers:** Adam Optimizer (for both Generator and Discriminator)  
- **Learning Rate:** 0.001
Used BCELoss for binary classification (real vs fake) and Adam optimizer for efficient training.
### 🔧 Code Snippet
```py
LR = 0.001
NUM_EPOCHS = 4000 
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters())
optimizer_generator = torch.optim.Adam(generator.parameters())
```


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
