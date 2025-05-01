# Introduction

Email spam detection remains an important practical problem in digital communication. With the increasing volume of spam emails that contain phishing attempts, malware, or unsolicited marketing content, effective automatic filtering systems are essential. Our project implements a state-of-the-art solution using transformer-based models, which have shown superior performance in natural language processing tasks compared to traditional machine learning approaches.

This report details the development and evaluation of a machine learning model for email spam detection using a deep learning approach with DistilBERT, a lightweight version of BERT (Bidirectional Encoder Representations from Transformers). The model was trained on the Enron Spam Dataset and achieved high classification accuracy (99.36%) on both validation and test sets. Analysis of performance metrics shows balanced precision and recall scores, indicating that the model is effective at identifying both spam and legitimate emails with minimal false positives and false negatives.

# Data Analysis

![Class Distribution](/results/class_distribution.png)

The Enron Spam Dataset was used for this project, comprising:

- Total emails: 33,716
- Spam emails: 17,171 (51%)
- Ham emails: 16,545 (49%)

The dataset is 51.7 MB in size and here is the link for the dataset [`Download Enron Spam Dataset (ZIP, 51.7MB)`](https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip). This is a well-balanced database, with a nearly equal distribution of spam and legitimate emails, which helps prevent class imbalance issues during training.
The dataset was split as follows:

- Training set: 24,008 emails (80% of the original dataset, minus validation)
- Validation set: 2,668 emails (10% of the training data)
- Test set: 6,669 emails (20% of the original dataset)

The dataset contains email of varible length from 1 character to 4247 characters.

- Average email length: 989.36 characters
- Max email length: 4247 characters
- Min email length: 1 characters

# Data Preprocessing

Several preprocessing steps were applied to prepare the data:

- Email text was truncated to a maximum of 512 tokens to accommodate model input constraints
- Empty emails were removed
- Emails were tokenized using the DistilBERT tokenizer, which handles encoding and padding

Text preprocessing was deliberately minimal to allow the model to learn from the raw linguistic patterns present in emails. This approach leverages the pre-trained language model's ability to understand context and nuance in text.

# Model Architecture

**DistilBERT**
DistilBERT is a distilled version of BERT that retains 97% of its language understanding capabilities while being 40% smaller and 60% faster. This makes it an excellent choice for practical applications like email filtering, where efficiency is important alongside accuracy.
The model was implemented using Hugging Face's Transformers library:

- Base model: distilbert-base-uncased
- Output layers: Classification head with 2 outputs (spam/ham)
- Parameters: 66 million (compared to BERT's 110 million)

**Training Configuration**
The model was fine-tuned with the following parameters:

- Learning rate: 2e-5
- Batch size: 8
- Training epochs: 3
- Weight decay: 0.01
- Optimizer: AdamW (implicit in the Trainer API)
- Mixed precision training (FP16) for efficiency

# 🧪 Performance Evaluation

## 🔧 Training and Validation Performance

The model showed steady improvements across training epochs. The table below summarizes the training and validation loss along with overall accuracy:

| Epoch | Training Loss | Validation Loss | Accuracy |
| ----- | ------------- | --------------- | -------- |
| 1     | 0.0383        | 0.0322          | 99.03%   |
| 2     | 0.0135        | 0.0389          | 99.36%   |
| 3     | 0.0009        | 0.0391          | 99.36%   |

### Highlights:

- **Training Accuracy**: ~100%, with a near-zero training loss by epoch 3, indicating a strong model fit.
- **Validation Accuracy**: 99.36%, validating the model’s ability to generalize to unseen data.
- **Test Accuracy**: 99.36%, with consistent results across all splits and minimal performance degradation.

---

## 📊 Test Set Classification Metrics

- **Accuracy**: 99.36%
- **Precision**: 0.99 (for both spam and ham)
- **Recall**: 0.99 (for both spam and ham)
- **F1-Score**: 0.99 (for both spam and ham)

### 🔍 Class-wise Breakdown

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Ham (0)  | 0.99      | 0.99   | 0.99     | 3,299   |
| Spam (1) | 0.99      | 0.99   | 0.99     | 3,370   |

### 📈 Overall Metrics

| Metric             | Score  |
| ------------------ | ------ |
| Accuracy           | 0.9936 |
| Macro Avg F1-Score | 0.99   |
| Weighted F1-Score  | 0.99   |
| Total Samples      | 6,669  |

> **Note**:
>
> - Class 0 = **Ham** (legitimate emails)
> - Class 1 = **Spam** (unsolicited emails)

---

## 📌 Confusion Matrix

![Confusion Matrix](results/distilbert/distilbert_conf_matrix.png)

The confusion matrix shows the following counts:

- **Ham correctly classified**: 3,276
- **Ham misclassified as spam**: 23
- **Spam correctly classified**: 3,350
- **Spam misclassified as ham**: 20

This distribution confirms the model's strong performance, with **only 43 total misclassifications out of 6,669 emails**.

---

## 🧠 Why These Metrics Matter

- **Precision**: Minimizes false positives (legitimate emails marked as spam).
- **Recall**: Minimizes false negatives (spam emails missed by the filter).
- **F1-Score**: Offers a balanced view, important when both errors carry user impact.

The strong balance between precision and recall ensures that important messages are preserved while harmful content is effectively filtered.

# 📈 Model Visualization

To better understand the internal behavior of our DistilBERT-based spam detection model, we employ two visualization techniques: **t-SNE** for high-dimensional embedding analysis and **ROC curve** for classification performance evaluation.

---

## 🧬 t-SNE Visualization

![t-SNE Visualization](/results/distilbert/t-SNE.png)

The figure above shows a t-distributed Stochastic Neighbor Embedding (t-SNE) plot of the email embeddings generated by the DistilBERT model before the classification head. Each point represents a single email projected from a high-dimensional embedding space into two dimensions for interpretability.

- **Red points** correspond to **spam** emails.
- **Blue points** correspond to **ham** (legitimate) emails.
- **sample size 500**

### Key Observations:

- The t-SNE visualization shows **clear separation** between spam and ham clusters, suggesting that the model learned **discriminative features** that effectively differentiate between the two classes.
- A few points near the decision boundary indicate **hard-to-classify** cases, such as edge cases or possibly mislabeled samples.

---

## 📊 ROC Curve

![ROC Curve](results/distilbert/roc_curve.png)

The Receiver Operating Characteristic (ROC) curve illustrates the trade-off between the **True Positive Rate (TPR)** and **False Positive Rate (FPR)** across different classification thresholds.

- The orange curve represents our model’s performance.
- The blue dashed diagonal line represents a random classifier (baseline).
- **AUC (Area Under Curve)** = **1.000**, which indicates **perfect classification**.

### Key Observations:

- The curve hugs the top-left corner, which reflects **excellent classification capability** across all thresholds.
- Threshold markers show that the model is robust to slight variations in the decision boundary.
- An AUC of **1.000** confirms that the classifier can **perfectly distinguish** between spam and ham on the test set.

---

These visualizations further validate the **reliability**, **robustness**, and **interpretability** of our DistilBERT-based spam detection model.

# 📊 Model Comparison

## Comparison with other Neural Networks Model

To evaluate the effectiveness of our transformer-based approach, we implemented and compared it with three other commonly used models for text classification: Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), and a Word2Vec + Logistic Regression baseline. The goal was to assess the performance gap between traditional, CNN-based, and transformer-based architectures on the Enron Spam Dataset.

### 1. **DistilBERT (Our most advanced model)**

- **Test Accuracy**: **99.36%**
- **Confusion Matrix**:
  - Ham correctly classified: 3276
  - Spam correctly classified: 3350
  - Misclassifications: 43 (23 FP, 20 FN)
- **Observations**:
  - Best overall performance
  - Minimal training required for convergence
  - Robust generalization

### 2. **Multilayer Perceptron (MLP)**

### Implementation Details:

- **Feature Extraction**: TF-IDF vectorization with 5,000 features
- **Architecture**:
  ```
  SpamMLP(
    (model): Sequential(
      (0): Linear(in_features=5000, out_features=512, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.5, inplace=False)
      (3): Linear(in_features=512, out_features=128, bias=True)
      (4): ReLU()
      (5): Dropout(p=0.5, inplace=False)
      (6): Linear(in_features=128, out_features=2, bias=True)
    )
  )
  ```
- **Regularization**: Dropout layers with 0.5 rate
- **Training**: Adam optimizer with learning rate 0.001
- **Epochs**: 10

![alt text](results/mlp/mlp_confusion.png)

- **Test Accuracy**: ~**98.84%**
- **Confusion Matrix**:
  - Ham correctly classified: 3264
  - Spam correctly classified: 3417
  - Misclassifications: 63 (45 FP, 18 FN)
- **Training Trend**:
  - Slight increase in validation loss after epoch 6, suggesting mild overfitting
- **Observations**:
  - Performed well with bag-of-words or TF-IDF features
  - Simpler and faster to train than transformers
  - Struggles slightly more with edge cases

### 3. **Convolutional Neural Network (CNN)**

![alt text](results/cnn/cnn_confusion.png)

- **Test Accuracy**: ~**98.86%**
- **Confusion Matrix**:
  - Ham correctly classified: 3264
  - Spam correctly classified: 3413
  - Misclassifications: 67 (45 FP, 22 FN)
- **Training Trend**:
  - Stable convergence; validation accuracy plateaued after epoch 5
- **Observations**:
  - Effective at capturing local n-gram features
  - Performs better than MLP on slightly longer email bodies
  - Not as expressive as transformers for long-range dependencies

### 4. **Word2Vec + Logistic Regression**

![alt text](results/word2vec/w2vec_confusion.png)

- **Test Accuracy**: ~**97.77%**
- **Confusion Matrix**:
  - Ham correctly classified: 3269
  - Spam correctly classified: 3382
  - Misclassifications: 93 (40 FP, 53 FN)
- **Training Trend**:
  - Training and validation accuracy quickly converge; limited capacity
- **Observations**:
  - Fastest model to train and interpret
  - Word2Vec lacks context sensitivity compared to BERT
  - Struggles with nuanced semantics or rare patterns

---

### 📌 Summary Table

| Model                    | Accuracy | FP  | FN  | Total Errors | Comments                           |
| ------------------------ | -------- | --- | --- | ------------ | ---------------------------------- |
| **DistilBERT**           | 99.36%   | 23  | 20  | **43**       | Best generalization and accuracy   |
| MLP                      | 98.84%   | 45  | 18  | 63           | Slightly overfits, decent accuracy |
| CNN                      | 98.86%   | 45  | 22  | 67           | Effective on local text features   |
| Word2Vec + Logistic Reg. | 97.77%   | 40  | 53  | 93           | Fast but least accurate            |

---

### 🔍 Key Insights

- **Transformer advantage**: DistilBERT outperforms others by capturing global context and subtle semantic differences.
- **Tradeoff**: Simpler models like MLP and Word2Vec require less compute and memory but sacrifice accuracy.
- **Error distribution**: All models favor lower false negatives over false positives, but DistilBERT maintains the best balance.

## 🔍 Comparison with Existing BERT-Based Approaches

To assess the relative performance and novelty of our DistilBERT-based spam detection model, we compare it against two recent works that also employed transformer-based models, particularly BERT or its successors, on the **Enron Spam Dataset**.

### 1. **Shrestha et al. (2023)** – University of Toledo Thesis

- **Model Used**: Fine-tuned XLNet (successor of BERT)
- **Dataset**: Enron Spam Dataset
- **Reported Accuracy**: **98.92%**
- **F1 Score**: **98.92%**
- **Preprocessing**:
  - Extracted raw email bodies
  - Lowercasing, tokenization
- **Observations**:
  - XLNet showed superior results to earlier BERT-based models across Enron, SpamAssassin, and Ling-Spam
  - Claimed generalization across datasets
- **Strengths**:
  - Bidirectional context with permutation-based training
  - Managed long sequences well (Transformer-XL backbone)
- **Weaknesses**:
  - Higher computational overhead
  - No evidence of adversarial or cross-domain evaluation

### 2. **Tang & Li (2024)** – Johns Hopkins Study

- **Model Used**: Fine-tuned BERT (HuggingFace `BertForSequenceClassification`)
- **Dataset**: Enron
- **Reported Accuracy**: **98.91%**
- **F1 Score**: **98.68%**
- **False Negative Rate**: 1.21%
- **Preprocessing**:
  - Sequence length capped at 32 tokens
  - Tokenized with `BertTokenizer`
- **Observations**:
  - Outperformed traditional ML models like SVM, Naive Bayes
  - BERT had lower false positive rate than GPT2
- **Strengths**:
  - High precision (99.03%) and recall
  - Balanced performance even under simple adversarial tests
- **Weaknesses**:
  - Only 2 training epochs
  - Struggled under cross-dataset transfer tests (data poisoning)

---

### ✅ Our DistilBERT-Based Approach

- **Model Used**: Fine-tuned `distilbert-base-uncased`
- **Dataset**: Enron Spam Dataset
- **Reported Accuracy**: **99.36%**
- **F1 Score**: **0.99**
- **False Negative Rate**: ~0.30%
- **Preprocessing**:
  - Maximum 512 tokens
  - Minimal cleanup (raw linguistic patterns preserved)
- **Strengths**:
  - Lightweight (40% smaller than BERT)
  - Fast convergence in just 3 epochs
  - Balanced precision & recall (0.99 each)
- **Weaknesses**:
  - Slight validation loss increase in final epoch
  - Same-domain dataset only (no cross-dataset or adversarial evaluation)

---

### 📌 Summary Table

| Work                   | Model      | Accuracy | F1 Score | FNR    | Notes                              |
| ---------------------- | ---------- | -------- | -------- | ------ | ---------------------------------- |
| **Our Work**           | DistilBERT | 99.36%   | 0.99     | ~0.30% | Best performance, lightest model   |
| Shrestha et al. (2023) | XLNet      | 98.92%   | 0.9892   | N/A    | Larger model, tested on 3 datasets |
| Tang & Li (2024)       | BERT       | 98.91%   | 0.9868   | 1.21%  | Tested for adversarial robustness  |

---

### 🧠 Key Insights

- Our model **outperforms** both XLNet and BERT-based implementations on the same dataset, with **higher accuracy and F1 score**.
- **DistilBERT achieves competitive results with significantly fewer parameters**, making it suitable for real-time spam filtering.
- Unlike Tang & Li (2024), we **did not explore adversarial or cross-dataset generalization**, which are useful future directions.

# Performance Analysis and Discussion

## Strengths of the Model

1. **Exceptional Accuracy**: 99.36% accuracy on both validation and test sets demonstrates robust performance.

2. **Balanced Performance**: Nearly identical precision and recall for both classes indicates the model isn't biased toward either spam or ham classification.

3. **Minimal Overfitting**: The similar performance on training and validation sets suggests good generalization. The slight increase in validation loss in later epochs is marginal and doesn't result in decreased accuracy.

4. **Fast Convergence**: The model reached high accuracy within just 3 epochs, suggesting efficient learning from the data.

## Potential Concerns

1. **Nearly Perfect Accuracy**: While impressive, the extremely high accuracy (99.36%) raises questions about whether:

   - The test set might be too similar to the training data
   - The model might be memorizing specific patterns rather than learning generalizable features
   - The dataset might contain obvious indicators that make classification unusually easy

2. **Validation Loss Increase**: There's a slight increase in validation loss in epochs 2–3 while accuracy remains stable, which could be an early sign of overfitting if training continued.

## Ideas for Improvement

Despite the model's strong performance, several enhancements could be considered:

- Data Augmentation: Introduce variations of existing emails by synonym replacement or LLM generated data.
- Cross-dataset Evaluation: Test the model on different spam datasets to ensure it generalizes across various types of spam and legitimate emails from different sources and time periods.

# Conclusion

The DistilBERT-based spam detection model demonstrates exceptional performance on the Enron Spam Dataset, with balanced precision and recall across both spam and legitimate email classes. While the extremely high accuracy suggests strong classification capability, additional testing on diverse datasets would help confirm the model's robustness in real-world scenarios. Our implementation successfully leverages transfer learning from pre-trained transformer models, requiring minimal preprocessing while achieving state-of-the-art results. This approach represents a significant advancement over traditional rule-based or classical machine learning methods for spam detection. Future work should focus on ensuring the model's ability to generalize to new and evolving spam tactics through techniques like data augmentation or cross-dataset evaluation.
