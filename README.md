# Keep Your Inbox Safe (KYIS): Transformer-Based Email Threat Detection

## Introduction
Email security threats have become increasingly sophisticated, outpacing many traditional detection mechanisms. Modern malicious emails encompass a wide range of attacks, including phishing, spam, malware-laden messages, scam/fraud attempts, and business email compromise (BEC) schemes. In response to these evolving risks, the objective of **Keep Your Inbox Safe (KYIS)** is to create an advanced neural network-based classification system that accurately identifies various types of email threats.

Rather than simply fine-tuning an existing transformer model (such as BERT), we will focus on **designing a custom transformer-inspired architecture** from scratch specifically tailored to the nuances of next-generation email threats. By introducing new structural elements and comparing them against state-of-the-art solutions, we aim to offer valuable insights into improving email security.

---

## High-Level Solution

### Custom Neural Network Architecture
KYIS will employ a from-scratch neural network design inspired by transformer architectures. Although transformers like BERT provide an excellent foundation for text processing, our goal is to build a custom model that thoroughly addresses the context-specific features of email data. To achieve this, we will experiment with different configurations—such as varying model depth, attention heads, and feed-forward layer sizes—to find the balance between classification accuracy and computational efficiency.

### Data Preprocessing
A robust data preprocessing phase ensures that our custom model receives high-quality inputs:

1. **Text Cleaning:** Removing extraneous symbols, HTML tags, and hidden characters.  
2. **Tokenization & Vectorization:** Converting raw text into sequences of tokens and numerical embeddings that are compatible with neural networks.  
3. **Feature Extraction:** Identifying relevant meta-features (sender addresses, header patterns, embedded URLs) that may boost detection performance.

### Multi-Class Classification
We will initially categorize emails into three major classes: **spam, and benign**. However, we will also design the model to easily incorporate additional malicious categories in the future—such as **phishing** and **BEC**. The ability to adapt to emerging threats is a priority, ensuring that KYIS remains effective as adversaries evolve their tactics.

### Comparison with State-of-the-Art Models
An integral part of our project involves benchmarking KYIS against established industry or research-level solutions. We plan to:
- **Fine-Tune and Evaluate** pre-trained models like BERT, RoBERTa, and T5 on the same dataset.  
- **Conduct Head-to-Head Comparisons** using consistent evaluation protocols and metrics.  
- **Identify Strengths and Weaknesses** of both KYIS and the competitor models to make further improvements.

### Evaluation and Validation
Measuring performance will involve commonly used metrics such as **accuracy, precision, recall, and F1-score**. To avoid overfitting, we will employ cross-validation strategies and keep an isolated test set for final evaluation. Additionally, we will compare KYIS against traditional spam filters and simpler machine learning classifiers to establish a performance baseline.

---

## Data Requirements
Ensuring high-quality, representative data is critical for building a reliable email classification model. We plan to divide our data into three subsets:

1. **Training Set (80%)** – Used to **learn** model parameters. This subset will contain a balanced distribution of phishing, spam, and benign emails where feasible.  
2. **Validation Set (10%)** – Used for hyperparameter tuning and early **performance checks**. This will ensure the model’s design choices generalize to data not used during training, mitigating overfitting.  
3. **Test Set (10%)** – Held out until the final phase to **evaluate** the model under realistic conditions.

### Potential Data Sources
- **OpenPhish** and **PhishTank**: Repositories with verified phishing emails.  
- **Enron Email Dataset**: A large corpus of legitimate corporate emails.  
- **SpamAssassin Public Corpus**: A well-known dataset for spam email detection.   
- **Manually Collected/Generated Samples**: High-value examples for specialized threats (e.g., spear-phishing & BEC, etc).

---

## Considerations and Challenges

1. **Custom Model Design Complexity**  
   Striking a balance between accuracy and computational efficiency is essential. Since we aim to classify emails in near real-time, the model must be optimized for both speed and accuracy. Additionally, the design must handle a diverse range of writing styles, file types, and languages.

2. **Feature Engineering**  
   While textual analysis is primary, metadata-based features (e.g., IP addresses, domain reputations, email header anomalies) can provide important cues. Identifying and integrating these features effectively could significantly enhance the model’s robustness.

3. **Handling Imbalanced Data**  
   Phishing emails are typically less frequent than spam or benign emails, which risks biasing the model toward majority classes. We will use techniques like **oversampling, undersampling, or synthetic data generation** to manage data imbalance.

4. **Adversarial Attacks & Evasion Techniques**  
   Malicious actors frequently update their strategies to bypass existing defenses. We will investigate adversarial training methods —i.e., introducing slightly altered phishing examples during training to build resilience against such threats.

5. **Real-Time Performance & Scalability**  
   Organizations often require instant email filtering. This necessitates us to explore optimizations like model pruning, weight quantization, or lighter architectures that can be deployed at scale (e.g., in a cloud environment).

---

## Next Steps

Our development of KYIS will be iterative, continually evolving to tackle challenges encountered during its lifecycle:

1. **Model Construction & Initial Training**  
   Implement a transformer-inspired network, tune hyperparameters, and establish a performance baseline using the training and validation sets.

2. **Data Augmentation & Threat Simulation**  
   Generate synthetic phishing emails to increase data variety and incorporate adversarial examples to fortify the model’s defenses.

3. **Benchmarking & Refinement**  
   Compare KYIS to state-of-the-art pre-trained transformers, simpler machine learning approaches, or existing research work in the literature. Use these insights to refine the architecture and model performance.

By proactively planning and implementing these stages, we aim to deliver a robust, scalable, and explainable system for classifying and managing diverse email threats. Through the combination of innovative architecture design and comparative testing with different models, we will demonstrate a deeper understanding and effective solution to modern email security challenges.

---

## Dataset Description

To effectively train and evaluate **Keep Your Inbox Safe (KYIS)**, we will use the **Spam Email Dataset**, a publicly available dataset from Kaggle. This dataset aggregates multiple well-known email repositories, providing a diverse and balanced representation of benign and spam emails.  

### Source Information  
- **Dataset Name:** Spam Email Dataset  
- **Download Link:** [Kaggle - Spam Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)
- **Reference to Original Collections:**
   - Enron Email Dataset: [CMU Enron Page](https://www.cs.cmu.edu/~enron/)
   - SpamAssassin Public Corpus: Provided by The Apache Software Foundation. [Link](https://spamassassin.apache.org/)
- **Associated Datasets Details:**  
  - **Enron and Ling Datasets:** Focus on the core textual content of phishing and legitimate emails.  
  - **CEAS, Nazario, Nigerian Fraud, and SpamAssassin Datasets:** Offer contextual metadata, such as sender and recipient information, timestamps, and spam classification.
  - **Dataset Details:** We will use the **phishing_email.csv** dataset, which consists of 82,486 email samples and 2 columns:  
      - **text_combined (string):** The full text of the email (subject + body).  
      - **label (integer):** Classification label:  
         - `1` = Spam/Phishing  
         - `0` = Benign (Non-Spam
- **Number of Emails:** 82,486  
- **Composition:**  
  - 42,891 spam emails  
  - 39,595 benign emails  

These sources contribute to a comprehensive dataset, covering a broad spectrum of spam tactics and legitimate communication styles.  
To ensure optimal model performance, we divide the dataset into three subsets:  

### Train/Validation/Test Split
To ensure reliable model development and evaluation, we divide the dataset into three subsets:

- Training Set (80%) → 66,000 emails

This large portion includes a broad mix of spam messages from diverse sources: Nigerian scams, phishing attempts from suspicious domains, and obfuscated spam. Benign emails include real corporate communications (e.g., Enron archives) and personal correspondences.
Because the training set is slightly imbalanced (~52% spam vs. 48% benign), we apply oversampling/undersampling techniques to avoid model bias.
- Validation Set (10%) → 8,250 emails

Contains a balanced mini-collection of the same email sources as the training set, plus a small portion of emails with newly encountered phrases, domains, or obfuscation methods. This design helps ensure our model is evaluated on patterns that partially overlap with but are not entirely identical to those in the training data.
These differences matter because modern spam evolves quickly—new linguistic tricks or scam strategies appear regularly. Evaluating on near-novel patterns helps us monitor the model’s robustness and generalization.
- Test Set (10%) → 8,250 emails

Provides a final, unbiased assessment of the model. Like the validation set, it includes a range of spam messages from multiple sources, as well as legitimate emails that mirror actual workplace communication scenarios.

- **Average Words Per Email:**  
  - Phishing Emails: ~120 words  
  - Benign Emails: ~180 words  
While phishing emails tend to be shorter and often contain urgent calls to action, benign emails vary in length, structure, and complexity.  

### Characterization of Email Samples  

#### Structure of Each Sample  
Each email sample in the dataset contains:
- Full Email Text (text_combined): A combination of the subject line and body text.
- Classification Label (label): Indicates whether the email is phishing (1) or benign (0). 

#### Key Features of Each Email Sample
- Phishing Emails
   May contain grammatical errors, fake links, and urgency tactics.
   Often lack personal details and attempt to deceive recipients.
- Benign Emails
   Contain structured, professional language.
   Often include formal greetings, clear subject lines, and well-formed sentences.

#### Email Content & Format  
- **Spam/Phishing Emails:**  
   - **Typical Traits:**
      * High urgency and threats (e.g., "Act Now!", "Immediate Action Required").
      * Includes deceptive URLs or requests for sensitive information.
      * May contain gibberish or obfuscated words to evade spam filters.
  - **Example Subject:** `"Your Account Has Been Compromised! Act Now!"`  
  - **Example Body:**  
    > *Dear User, we noticed suspicious activity on your bank account. Please verify your identity immediately by clicking the link below.*  

- **Benign Emails:**  
  - Well-structured, professional language.
  - Contextually relevant and often work-related.
  - Does not contain deceptive links or suspicious requests.
  - **Example Subject:** `"Meeting Reminder: Q3 Financial Report"`  
  - **Example Body:**  
    > *Hi John, please find attached the Q3 financial report for review. Let's discuss in our next meeting.*   
