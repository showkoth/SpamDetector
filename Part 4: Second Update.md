# Introduction
Email spam detection remains an important practical problem in digital communication. With the increasing volume of spam emails that contain phishing attempts, malware, or unsolicited marketing content, effective automatic filtering systems are essential. Our project implements a state-of-the-art solution using transformer-based models, which have shown superior performance in natural language processing tasks compared to traditional machine learning approaches.

This report details the development and evaluation of a machine learning model for email spam detection using a deep learning approach with DistilBERT, a lightweight version of BERT (Bidirectional Encoder Representations from Transformers). The model was trained on the Enron Spam Dataset and achieved high classification accuracy (99.36%) on both validation and test sets. Analysis of performance metrics shows balanced precision and recall scores, indicating that the model is effective at identifying both spam and legitimate emails with minimal false positives and false negatives.

# Data Analysis
The Enron Spam Dataset was used for this project, comprising:

- Total emails: 33,716
- Spam emails: 17,171 (51%)
- Ham emails: 16,545 (49%)

This dataset is well-balanced, with a nearly equal distribution of spam and legitimate emails, which helps prevent class imbalance issues during training.
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

#  Performance Metrics
**Training and Validation Performance**
As shown below table, the model showed consistent improvement across training epochs:

| Epoch | Training Loss | Validation Loss | Accuracy   |
|-------|----------------|------------------|------------|
| 1     | 0.038300       | 0.032217         | 0.990255   |
| 2     | 0.013500       | 0.038935         | 0.993628   |
| 3     | 0.000900       | 0.039140         | 0.993628   |

It achieved excellent results on both training and evaluation sets. Below we summarize the key classification metrics, including accuracy, precision, recall, and F1-score, and discuss their significance for the spam detection task.
- **Training Accuracy:** ~100%. The model nearly perfectly fit the training data. By the end of training (epoch 3), the training loss had dropped to 0.0009​, indicating almost zero classification errors on the training set. In terms of raw counts, this implies essentially all ~24,008 training emails were classified correctly, with virtually no misclassifications.
- **Validation Accuracy:** 99.36%. On the held-out validation set (about 2,668 emails), the model achieved 99.36% accuracy, correctly classifying roughly 2,651 out of 2,668 examples. Only a small number of emails were misclassified. This high validation accuracy closely matched training performance, suggesting the model generalizes well to unseen data.
- **Test Accuracy:** 99.36%. Finally, on the independent test set (6,669 emails set aside for final evaluation), the model also attained 99.36% accuracy​. It correctly predicted the label for 6,626 of 6,669 emails, with only 43 emails (≈0.64%) incorrectly classified as either false positives or false negatives. This consistency between validation and test accuracy indicates stable generalization.

**Test Set Performance**
The model achieved exceptional performance on the test set:

- Accuracy: 99.36%
- Precision: 0.99 for both classes
- Recall: 0.99 for both classes
- F1-score: 0.99 for both classes

**Detailed Metrics Analysis**
The classification report shows balanced performance across both classes:
precision    recall  f1-score   support

           0       0.99      0.99      0.99      3299
           1       0.99      0.99      0.99      3370

    accuracy                           0.99      6669
   macro avg       0.99      0.99      0.99      6669
weighted avg       0.99      0.99      0.99      6669

Where:

Class 0: Ham (legitimate emails)
Class 1: Spam emails

These metrics are particularly important in spam detection because:

- Precision: The proportion of predicted spam emails that are actually spam. High precision means few legitimate emails are misclassified as spam, which is critical to avoid losing important communications.
- Recall: The proportion of actual spam emails that are correctly identified. High recall means most spam is caught by the filter, protecting users from harmful content.
- F1-score: The harmonic mean of precision and recall, providing a balanced measure of the model's performance. A high F1-score indicates that the model maintains both high precision and high recall.

**Confusion Matrix**
The confusion matrix provides a visualization of the model's classification performance:
![Confusion Matrix](results/distilbert/distilbert_conf_matrix.png)
The confusion matrix shows minimal misclassifications in either direction, confirming the model's strong performance.

For spam detection, balanced metrics are crucial. While accuracy provides an overall measure of performance, precision and recall are particularly important in this domain:

1. False Positives (legitimate emails classified as spam) can lead to missed important communications, potentially resulting in significant negative consequences. High precision minimizes this risk.
2. False Negatives (spam classified as legitimate) expose users to potentially harmful content. High recall ensures most spam is captured.

The F1-score was selected as a primary evaluation metric because it balances these concerns, ensuring the model performs well in both aspects. The confusion matrix further helps visualize the distribution of errors, confirming that misclassifications are minimal and evenly distributed between classes.

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
- t-SNE Visualization: We can analyze the model embedding using t-SNE to understand the model better.
- Cross-dataset Evaluation: Test the model on different spam datasets to ensure it generalizes across various types of spam and legitimate emails from different sources and time periods.


# Conclusion
The DistilBERT-based spam detection model demonstrates exceptional performance on the Enron Spam Dataset, with balanced precision and recall across both spam and legitimate email classes. While the extremely high accuracy suggests strong classification capability, additional testing on diverse datasets would help confirm the model's robustness in real-world scenarios. Our implementation successfully leverages transfer learning from pre-trained transformer models, requiring minimal preprocessing while achieving state-of-the-art results. This approach represents a significant advancement over traditional rule-based or classical machine learning methods for spam detection. Future work should focus on ensuring the model's ability to generalize to new and evolving spam tactics through techniques like data augmentation or cross-dataset evaluation.
