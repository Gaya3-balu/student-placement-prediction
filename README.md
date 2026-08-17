# Student Placement Prediction

A machine learning model that predicts whether a student will be placed in a job, based on academic performance, technical skills, and extracurricular activity — trained on a 100,000-row dataset.

## Tools
Python, pandas, numpy, scikit-learn (Logistic Regression), Google Colab

## Features Used
Branch, college tier, CGPA, backlogs, coding skills, DSA score, aptitude score, communication skills, ML knowledge, system design, internships, projects count, certifications, hackathons, open-source contributions, extracurriculars

## Process
- Cleaned data (verified no missing values), dropped an irrelevant salary column
- Encoded categorical columns (branch, college tier) using LabelEncoder
- Split data into training and test sets (80/20, stratified by placement outcome)
- Trained a Logistic Regression model

## Results
- Train accuracy: ~69.8%
- Test accuracy: ~69.8%
- Consistent train/test accuracy indicates no overfitting

## Files
`student_placement.ipynb` — full notebook: data cleaning, encoding, model training, and prediction on new data
