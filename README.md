# Hand Gesture Classification

**Christian Braz, Junior Ovince, Rahul Seti**  
George Washington University — Department of Data Science

---

## Abstract

Hand gesture recognition is the process of recognizing meaningful expressions of form and motion by a human involving only the hands. There are plenty of applications where hand gesture recognition can be applied for improving control, accessibility, communication, and learning. In this work, we present our results in classifying twenty-six hand signs for the letters of the alphabet. We conduct experiments using different convolutional neural network architectures and report the performance of each of them.

**Keywords:** gesture recognition, machine learning, convolutional neural networks, deep learning

---

## Introduction

Hand gestures provide a complementary modality for expressing ideas. They can be used in many different contexts to augment human-machine interaction — for instance, to help deaf people communicate, send commands to an intelligent house system, and so on. There are mainly two ways to capture such interaction: via **Data-Glove** and the **vision approach**. The Data-Glove based approach collects data from sensors attached to a glove mounted on the user's hand. It is accurate and captures only the necessary information, minimizing the amount of data. However, due to the required hardware, it is inconvenient and even infeasible in many scenarios. The vision approach has the advantage of hardware simplicity, but requires a large amount of image data to be processed in an attempt to produce a solution that is insensitive to adverse conditions like lighting, background, subject, and camera variations.

The **American Sign Language (ASL)** is a visual language that incorporates gestures, facial expressions, and body movements. Hand signs are the foundation of the language. Many signs are iconic, meaning the sign uses a visual image that resembles the concept it represents. The **American Manual Alphabet (AMA)** is a manual alphabet that augments the vocabulary of ASL. When using hand signs for letters to spell out a word, this is called "fingerspelling" — in AMA, this is one-handed. The AMA is composed of 26 hand positions designating letters from A to Z, and 10 positions representing the digits 0 to 9.

In this work, we address the vision-based solution to classifying the 26 hand gesture positions for fingerspelling the letters of the alphabet. The paper is organized as follows: **Related Work** presents a review of general hand gesture recognition approaches and works related to sign language recognition. **Literature Review** describes the main CNN models used: LeNet5, AlexNet, VGG Net, and GoogleLeNet. **Methods** details the training approach, dataset, and models. **Conclusion** discusses findings and future work.

---

## Related Work

Hand gesture recognition without extra wearable devices has been the target of many research efforts. Its applicability spans video games, augmented reality, intelligent house system control, and 3D interaction scenarios. Trigueiros, Ribeiro, and Reis (2012) conducted a comparative study of four algorithms — k-Nearest Neighbour (k-NN), Naïve Bayes (NB), Artificial Neural Network (ANN), and Support Vector Machines (SVM) — for static gesture recognition, concluding that ANN delivered the best performance.

Strezoski et al. (2018) highlight how specialized hardware such as GPUs boosted deep learning architectures, particularly CNNs, which have proven superior for image processing problems. In their paper, they evaluate efficient CNN architectures designed by Google, IBM, and Microsoft for identifying hand signs from the Marcel dataset.

Garcia and Viesca (2016) propose an ASL fingerspelling translator based on CNN, building on the pre-trained GoogleLeNet via transfer learning to develop a real-time ASL recognition system that translates video of a user's ASL signs into text. Bheda and Radpour (2017) tackle automatic sign language recognition using depth-sensing images that capture contour and depth information beyond regular pixel data.

---

## Literature Review

### Convolutional Neural Network

A Convolutional Neural Network (CNN) consists of a number of convolutional and pooling layers, optionally followed by fully connected layers. Convolution layers compute the inner product of a linear filter and the underlying image portion, followed by a nonlinear activation function at every local portion of the input. The resulting outputs are called **feature maps**. Its architecture is designed to exploit the 2D structure of an input image.

The input to a convolutional layer is an \(m \times m \times r\) image, where \(m\) is the height and width and \(r\) is the number of channels. A RGB image has three channels (Red, Green, Blue), each a 2D matrix of pixel values in the range 0–255. A grayscale image has one channel. The convolutional layer has \(k\) filters (or kernels) of size \(n \times n \times q\) which convolve the image to produce \(k\) feature maps of size \(m - n + 1\). Each feature map is then processed by a **pooling operator** that reduces dimensionality while retaining the most important information. Higher-level layers are typically fully connected layers (traditional MLP) that classify the input image using a softmax activation function.

### The LeNet Architecture

In the seminal 1994 paper, Yann LeCun defined the first CNN — **LeNet5**. It demonstrated that image features are distributed across the entire image, and that convolutions with learnable parameters are an effective way to extract features at multiple locations. LeNet5 established that images are highly spatially correlated, making pixel-by-pixel input to large MLPs suboptimal.

Designed for handwritten digit recognition, LeNet5 processes a 32×32×1 image through:
1. Six 5×5 filters with stride 1 → 28×28×6 feature map
2. Average pooling (filter 2, stride 2) → 14×14×6
3. Sixteen 5×5 filters + pooling → 5×5×16
4. Fully connected layer: 400 → 120 neurons
5. Fully connected layer: 120 → 84 neurons
6. Output layer: 84 → 10 neurons (digit classes)

### AlexNet

AlexNet, winner of ILSVRC-2010 (ImageNet Large Scale Visual Recognition Challenge), built on LeNet's insights to create a much larger network capable of learning more complex features. It employed 11×11, 5×5, and 3×3 convolutions, max pooling, dropout, and ReLU activations after every convolutional and fully-connected layer. AlexNet was trained on two Nvidia GeForce GTX 580 GPUs, enabling the use of larger datasets and higher-resolution images.

### VGG Net

Simonyan and Zisserman (2014) proposed increasing network depth using much smaller filter sizes:

> *"We fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small (3×3) convolution filters in all layers. Max-pooling is performed over a 2×2 pixel window, with stride 2."*

Stacking three 3×3 convolutional layers without pooling achieves an effective receptive field equivalent to a single 7×7 filter, with the advantage of three non-linear rectification layers instead of one. VGG Net trains over 224×224 RGB images using architectures of 11 to 19 layers and 133 to 144 million parameters, claiming to outperform GoogleLeNet in single-network classification accuracy.

### GoogleLeNet

GoogleLeNet, winner of ILSVRC 2014 with a top-5 error rate of **6.67%** (close to human-level performance), is a 22-layer deep architecture inspired by the **Network In Network (NIN)** idea.

NIN replaces the generalized linear model (GLM) used in conventional CNN filters with a multilayer perceptron (MLP) — a universal function approximator. Feature maps are obtained by sliding the MLP over the input similarly to CNN. The NIN method can be viewed as additional 1×1 convolutional layers:

> *"The cross channel parametric pooling layer is also equivalent to a convolution layer with 1×1 convolution kernel."*

GoogleLeNet uses NIN to create **Inception modules** — stacks of feature maps using 1×1 convolutions to compute reductions before the expensive 3×3 and 5×5 convolutions. This allows increasing network depth and width without a significant computational penalty. The name GoogleLeNet is a homage to Yann LeCun's pioneering LeNet 5.

---

## Methods

The experimental methodology is divided into three phases: **pretraining**, **training**, and **post-training**.

### A. Pretraining

Data preprocessing included two main components:

- **Normalization:** The dataset was loaded using PyTorch's `ImageFolder` package and normalized by 0.5 to scale all values to the range \([-1, +1]\).
- **Dataset split:** Images were split into train, validation, and test sets at a **70/15/15** ratio using scikit-learn's train/test split module, keeping all 29 classes balanced across the splits.

Three CNN architectures were selected — a custom Baseline, LeNet, and AlexNet — chosen because the dataset does not appear to require highly complex architectures.

### Network Architecture Summary

| | Baseline | LeNet | AlexNet |
|---|---|---|---|
| **Conv layers** | 2 conv (5,5) | 2 conv (5,5) | 5 conv (11,5,3,3,3) |
| **Fully connected layers** | 1 (250 neurons) | 2 (120, 84 neurons) | 2 (4096, 4096) |
| **Output activation** | LogSoftmax | Linear | Linear |
| **Estimated parameters** | 11M | 4.2M | 103M |
| **Estimated feature maps** | 230 | 114 | 250k |

### B. Training

**Performance index:** Cross-entropy loss, suitable for multiclass probability distribution estimation.

**Optimizer:** Adam optimizer — a robust variant of stochastic gradient descent.

**Regularization and convergence techniques:**
- Xavier Normal initialization for convolutional layers
- Learning rate scheduler (updates when losses plateau)
- Dropout nodes to prevent overfitting
- Early stopping

All models were trained for a total of **20 epochs** as the initial benchmark.

### C. Post Training

Post-training evaluation used the following classic metrics: confusion matrix, precision, recall, F-score, AUC, and ROC curve. Error statistics (mean, std) and processing time were also recorded to assess overall model quality. All models were run using **PyTorch** and **TensorBoard**.

---

## Results

### 1. Baseline

The baseline model performs well up to the **10th epoch**, after which it stops improving and starts to overfit. Results comparing 20-epoch and early-stopped runs are shown below.

| Metrics | Initial (20 epochs) | Final (10 epochs) |
|---|---|---|
| Precision | 0.99595 | 0.9916846 |
| Recall / Accuracy | 0.99593 | 0.9916475 |
| F-score | 0.99594 | 0.9916470 |
| AUC | All > 0.9999 | All > 0.9998 |
| Processing Time | 36 minutes | 19 minutes |
| Error (std) | 0.000649517 | 0.0007859 |
| Error (mean) | 0.000641096 | 0.00096642 |
| LR updated at epoch | 12 and 18 | None |
| Stopped at epoch | 20 | 10 |

### 2. LeNet

The LeNet model tends to overfit at the **4th epoch**. Comparing 20-epoch and 4-epoch runs confirms that 4 epochs is the optimal setting.

| Metrics | Initial (20 epochs) | Final (4 epochs) |
|---|---|---|
| Precision | 0.996112 | 0.976701 |
| Recall / Accuracy | 0.996091 | 0.975785 |
| F-score | 0.996091 | 0.975852 |
| AUC | All > 0.9801 | All > 0.9603 |
| Processing Time | 30 minutes | 7 minutes |
| Error (std) | 0.00081509 | 0.0008129 |
| Error (mean) | 0.0009158 | 0.002219 |
| LR updated at epoch | None | None |
| Stopped at epoch | 20 | 4 |

### 3. AlexNet

AlexNet is designed for large-scale problems with hundreds or thousands of classes. Given that LeNet achieves strong performance with only two convolutional layers, AlexNet is over-specified for this dataset and was run for only **5 epochs** due to its high computational cost (45 minutes). Even then, it exhibits overfitting.

| Metrics | Initial (5 epochs) |
|---|---|
| Precision | 0.989630 |
| Recall / Accuracy | 0.989272 |
| F-score | 0.989259 |
| AUC | All > 0.9949 |
| Processing Time | 44 minutes |
| Error (std) | 0.0004457 |
| Error (mean) | 0.0007320 |
| LR updated at epoch | None |
| Stopped at epoch | 5 |

### 4. Model Testing

Both the Baseline and LeNet models were evaluated on the held-out test set:

| Metrics | Baseline (10 epochs) | LeNet (4 epochs) |
|---|---|---|
| Precision | 0.9902369 | 0.9794684 |
| Recall / Accuracy | 0.9901915 | 0.9786973 |
| F-score | 0.9901984 | 0.9787579 |
| AUC | All > 0.9998 | All > 0.9596 |
| Processing Time | 18 minutes | 7 minutes |
| Error (std) | 0.0007808 | 0.000858553 |
| Error (mean) | 0.000998 | 0.00224975 |
| Classes with lowest accuracy | S, A, B, del | G, P, Space, B |

The **Baseline model** achieves higher accuracy; the **LeNet model** classifies at ~98% accuracy in only 7 minutes. The accuracy/speed trade-off is a classic data science decision: LeNet is preferable for real-time translation applications, while Baseline suits less time-sensitive tagging applications.

---

## Conclusion

This project compared different CNN architectures for classifying 26 American Sign Language alphabet hand gestures. Key findings:

- A **two-convolutional-layer network** is sufficient for this dataset — a complex architecture like AlexNet is unnecessary and computationally expensive.
- The **Baseline model** outperforms LeNet in accuracy, while **LeNet** classifies at ~98% accuracy nearly 2.5× faster.
- A **combination or ensemble** of the two architectures could improve overall performance.
- Very little hyperparameter tuning was performed; adjusting the number of neurons, batch size, and weight regularization could yield further improvements.

---

## References

Bheda, V., & Radpour, D. (2017). Using deep convolutional networks for gesture recognition in American Sign Language. *CoRR, abs/1710.06836*. Retrieved from http://arxiv.org/abs/1710.06836

Garcia, B., & Viesca, S. (2016). Real-time American Sign Language recognition with convolutional neural networks. In *Convolutional Neural Networks for Visual Recognition*. Stanford.

Lin, M., Chen, Q., & Yan, S. (2013). Network in network. *CoRR, abs/1312.4400*. Retrieved from http://arxiv.org/abs/1312.4400

Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *CoRR, abs/1409.1556*. Retrieved from http://arxiv.org/abs/1409.1556

Strezoski, G., Stojanovski, D., Dimitrovski, I., & Madjarov, G. (2018). Hand gesture recognition using deep convolutional neural networks. In G. Stojanov & A. Kulakov (Eds.), *ICT Innovations 2016* (pp. 49–58). Cham: Springer International Publishing.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S. E., Anguelov, D., … Rabinovich, A. (2014). Going deeper with convolutions. *CoRR, abs/1409.4842*. Retrieved from http://arxiv.org/abs/1409.4842

Trigueiros, P., Ribeiro, F., & Reis, L. P. (2012, June). A comparison of machine learning algorithms applied to hand gesture recognition. In *7th Iberian Conference on Information Systems and Technologies (CISTI 2012)* (pp. 1–6).
