# KYIS: First Update Report

In the first update, I tried to benchmark spam detection task using the state-of-the-art neural networks models which will be used as the baseline models. I implemented the word2vec, MLP, and CNN based models from scratch to see how these models work for email classification task.

## Dataset

For this phase, we have used the Enron email spam dataset, which contains approximately 33,000 labeled emails. The dataset provides a comprehensive collection of both spam and legitimate emails in a real-world context, making it ideal for training and evaluating spam detection models.

Dataset statistics:
- Total emails: 33,716
- Spam emails: 17,171 (51%)
- Ham emails: 16,545 (49%)

The balanced nature of this dataset helps in training models that don't suffer from class imbalance issues.

## Model Implementations and Results

### 1. MLP with TF-IDF Vectorization

Our first approach utilizes a Multi-Layer Perceptron (MLP) with TF-IDF vectorization for feature extraction.

#### Implementation Details:
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

#### Results:
The MLP model achieved excellent performance, with the following metrics:
- **Accuracy**: 98.6% on test set
- **Confusion Matrix**: 
  - True Negatives (Ham correctly classified): 3269
  - False Positives (Ham misclassified as spam): 40
  - False Negatives (Spam misclassified as ham): 53
  - True Positives (Spam correctly classified): 3382

### 2. CNN with Word Embeddings

Our second approach employs a Convolutional Neural Network (CNN) designed specifically for text classification.

#### Implementation Details:
- **Architecture**:
  ```
  TextCNN(
    (embedding): Embedding(10000, 100, padding_idx=0)
    (convs): ModuleList(
      (0): Conv2d(1, 100, kernel_size=(3, 100), stride=(1, 1))
      (1): Conv2d(1, 100, kernel_size=(4, 100), stride=(1, 1))
      (2): Conv2d(1, 100, kernel_size=(5, 100), stride=(1, 1))
    )
    (fc): Linear(in_features=300, out_features=2, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  ```
- **Model Size**: 1,120,902 trainable parameters
- **Vocabulary**: 10,000 most common words
- **Embedding Dimension**: 100
- **Filter Sizes**: 3, 4, and 5 with 100 filters each
- **Training**: Adam optimizer with learning rate 0.001
- **Epochs**: 10

#### Results:
The CNN model achieved outstanding performance:
- **Accuracy**: 98.9% on test set
- **Confusion Matrix**:
  - True Negatives (Ham correctly classified): 3264
  - False Positives (Ham misclassified as spam): 45
  - False Negatives (Spam misclassified as ham): 22
  - True Positives (Spam correctly classified): 3413

### 3. Word2Vec with Neural Network Classifier

Our third approach leverages pre-trained word embeddings from Word2Vec to create semantic document representations.

#### Implementation Details:
- **Word Embeddings**: Custom-trained Word2Vec model on our corpus
- **Embedding Dimension**: 100
- **Word2Vec Training**: Skip-gram model with window size 5
- **Vocabulary Size**: 68,604 words
- **Document Vector Shape**: (24274, 100)
- **Classifier Architecture**:
  ```
  Word2VecClassifier(
    (model): Sequential(
      (0): Linear(in_features=100, out_features=128, bias=True)
      (1): ReLU()
      (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Dropout(p=0.3, inplace=False)
      (4): Linear(in_features=128, out_features=64, bias=True)
      (5): ReLU()
      (6): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Dropout(p=0.3, inplace=False)
      (8): Linear(in_features=64, out_features=2, bias=True)
    )
  )
  ```
- **Model Size**: 21,698 trainable parameters
- **Training**: Adam optimizer with learning rate scheduling
- **Epochs**: 20

#### Results:
The Word2Vec model also achieved impressive performance:
- **Accuracy**: 98.07% on test set
- **Confusion Matrix**:
  - True Negatives (Ham correctly classified): 3264
  - False Positives (Ham misclassified as spam): 45
  - False Negatives (Spam misclassified as ham): 18
  - True Positives (Spam correctly classified): 3417

## Performance Comparison

| Model | Accuracy | Parameters | Training Behavior | Strengths | Weaknesses |
|-------|----------|------------|-------------------|-----------|------------|
| MLP with TF-IDF | 98.6% | ~2.6M | Steady convergence | Simple, fast training | Word order ignorance |
| CNN | 98.9% | 1.12M | Rapid initial convergence | Best accuracy, pattern recognition | Complex architecture |
| Word2Vec | 98.07% | 21.7K | Potential overfitting | Smallest model, semantic understanding | Validation loss increases |

All three models provide strong classification performance, with the CNN slightly outperforming the others in raw accuracy. The Word2Vec approach stands out for its extremely compact model size (only ~2% of the CNN's parameter count) while still delivering competitive accuracy.

## Learning Curve Analysis

The learning curves for each model reveal important insights into their training dynamics:

### MLP Model
- Training and validation losses decrease consistently throughout training
- Both training and validation accuracy reach plateaus around 98-99%
- No significant gap between training and validation metrics, suggesting good generalization

### CNN Model
- Very rapid initial decrease in loss and increase in accuracy
- After epoch 4, the training accuracy continues to improve while validation accuracy plateaus
- Small divergence between training and validation accuracy after epoch 6, but not concerning

### Word2Vec Model
- Training loss decreases to near zero while validation loss increases after epoch 3
- Growing gap between training accuracy (approaching 100%) and validation accuracy (around 98.6%)
- Clear signs of overfitting despite regularization techniques like dropout and batch normalization

All models show excellent precision and recall for both classes, but Word2Vec has a slight tendency to misclassify spam as legitimate, which could be more problematic in a real-world application (missing spam is generally worse than incorrectly flagging legitimate emails).

## Current Challenges

### 1. Overfitting in Word2Vec Model
Despite having significantly fewer parameters, the Word2Vec model shows clearer signs of overfitting than the other approaches. The increasing validation loss suggests that the model is memorizing the training data rather than learning generalizable patterns. We need to:
- Explore stronger regularization techniques
- Investigate alternative document representation methods beyond simple averaging
- Consider early stopping based on validation loss

### 2. Model Size vs. Performance Trade-offs
Our implementations span three orders of magnitude in parameter count:
- Word2Vec: ~22K parameters
- CNN: ~1.1M parameters
- MLP: ~2.6M parameters (due to 5000-dimensional input)

While the CNN achieves the highest accuracy, its size might be prohibitive for certain deployment scenarios. We need to explore model compression techniques or develop hybrid approaches that maintain accuracy with fewer parameters.

### 3. Handling Adversarial Examples
None of our current implementations are explicitly designed to handle adversarial examples—emails crafted to evade detection. For example:
- Spam emails that intentionally misspell common trigger words
- Messages that hide spam content within legitimate-looking text
- Techniques that insert invisible characters to break pattern recognition

### 4. Feature Engineering Limitations
The TF-IDF approach requires manual feature selection (max_features=5000), potentially losing important information. The CNN and Word2Vec models are limited by fixed vocabulary sizes and embedding dimensions. We need more adaptive feature selection methods.

### 5. Interpretability
While our models achieve high accuracy, they provide limited insights into why specific emails are classified as spam or legitimate. This lack of interpretability could make it difficult to:
- Debug misclassifications
- Explain decisions to users
- Adapt to new spam techniques

## Next Steps (Phase 2)

1. **Advanced Models**: Implement BERT, RNN, and LSTM models for comparison with our current approaches.

2. **Addressing Overfitting**: Explore regularization techniques beyond dropout and batch normalization, such as:
   - Label smoothing
   - Mixup augmentation
   - Weight decay optimization

6. **Adversarial Training**: Introduce adversarial examples during training to improve robustness.

7. **Dynamic Vocabulary**: Implement techniques to handle out-of-vocabulary words and adapt to new spam patterns.

8. **Cross-Domain Evaluation**: Test models on different email sources to evaluate generalization capabilities.

## Conclusion

Our Phase 1 implementations demonstrate that even relatively simple neural network architectures can achieve high accuracy in email spam detection, with all three approaches achieving over 98% accuracy on the test set.

The CNN architecture offers the best overall performance with an accuracy of 98.9%, likely due to its ability to capture important n-gram patterns regardless of their position in the email. However, the Word2Vec approach achieves impressive results with a dramatically smaller model (21.7K vs. 1.12M parameters), making it potentially more suitable for resource-constrained environments.

The learning curves and confusion matrices provide valuable insights into model behavior, highlighting areas for improvement in Phase 2. Particularly concerning is the overfitting tendency of the Word2Vec model, which will require additional regularization strategies.

In Phase 2, we will focus on implementing more advanced architectures like BERT, RNN, and LSTM, while also addressing the challenges identified in this report. We will pay particular attention to the trade-off between model size and performance, seeking to develop models that are both accurate and efficient.