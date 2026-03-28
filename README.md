# NLP_Entity_Classification
Legal Entity Classification (Person vs. Company) for Insurance Data Quality using Character-Level Deep Learning (Char-CNN).
# Insurance MDM: Legal Entity Classification via Deep Learning
**Context:** In insurance databases, corrupted or poorly entered policyholder names (missing legal status like SARL/EURL) falsify risk mutualization and premium pricing.  
**Solution:** This project solves this data quality issue by deploying a Character-Level Convolutional Neural Network (Char-CNN) built with TensorFlow/Keras. By masking obvious keywords during training to prevent data leakage, the model learns the deep latent structure of North African names and corporate acronyms, achieving a 94%+ F1-Score on messy production data.
