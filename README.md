# Multimodal Information Retrieval on Amazon Reviews Dataset

## Overview

This project involves analyzing the similarity between text reviews and images from an Amazon reviews dataset. The dataset consists of reviews with associated images. The goal is to process and analyze these data points to find the most similar reviews based on both text and image features.

## Project Structure

1. **Loading and Preprocessing the Dataset**
   - **Dataset Repository**: The dataset is cloned from a Git repository.
   - **Data Import**: The dataset (`A2_data.csv`) is imported using pandas, resulting in a DataFrame with 1000 rows.
   - **Data Cleaning**: 
     - Added a header for the 'Review Id' column.
     - Dropped rows with empty values, resulting in 999 non-null rows.
     - Extracted the 'images' column into a new DataFrame, which expanded to 1647 rows due to repeated image links.
     - Saved the expanded DataFrame as a CSV file for clarity.

2. **Image Extraction**
   - **Preprocessing**: Used InceptionV3 for image feature extraction, applying preprocessing steps suitable for the dataset.
   - **Normalization**: Normalized the extracted image features by their mean and standard deviation.
   - **Error Handling**: Identified erroneous image links and removed the corresponding review IDs from the dataset to prevent them from affecting text processing.
   - **Saving Features**: Saved the normalized image features using the pickle module.

3. **Text Extraction**
   - **Term Frequency (TF)**: Calculated TF for each word in the 'Cleaned_Review' column.
   - **Inverse Document Frequency (IDF)**: Calculated IDF for each word across all documents.
   - **TF-IDF Calculation**: Combined TF and IDF scores to compute TF-IDF for each word in each document.
   - **Saving Results**: Saved the IDF dictionary and TF-IDF scores using pickle.

4. **Input Handling and Feature Flattening**
   - **Image Features**: Flattened the 3D array of normalized image features into 1D arrays.
   - **Text Features**: Generated the TF-IDF vector for the input review text using the saved IDF dictionary.

5. **Cosine Similarity Calculation**
   - **User Input**: Accepted the number of top-ranked reviews to return.
   - **Text Similarity**: Calculated cosine similarity between the TF-IDF vectors of reviews and the input text.
   - **Image Similarity**: Calculated cosine similarity between the normalized image features of reviews and the input image.
   - **Composite Similarity**: Computed the average of text-based and image-based cosine similarity scores.
   - **Ranking**: Selected the top 'n' reviews based on their similarity scores.

## Usage

1. **Clone the Repository**: Clone the dataset repository into your Jupyter Notebook environment.
2. **Load and Preprocess Data**: Use pandas to load the dataset and perform the necessary preprocessing steps.
3. **Extract Image Features**: Use InceptionV3 to preprocess and normalize image features.
4. **Calculate TF-IDF**: Implement functions to calculate TF, IDF, and TF-IDF for text reviews.
5. **Flatten Features**: Flatten image features and generate TF-IDF vectors for input text.
6. **Compute Similarity**: Calculate cosine similarities and determine the most similar reviews.
7. **Save Results**: Save the final results and any intermediate data using pickle.

## File Structure

- `CSE508_Winter2024_A2_Dataset\A2_Data.csv`: Original dataset file with 1000 rows.
- `expanded_images.csv`: Expanded dataset with individual and repeated image links (1647 rows).
- `idf_dict.pkl`: Pickle file containing the IDF dictionary.
- `tfidf_scores.pkl`: Pickle file containing the TF-IDF scores for each document.
- `normalized_image_features.pkl`: Pickle file containing the normalized image features.

## Requirements

- Python 3.x
- Jupyter Notebook
- Pandas
- Numpy
- TensorFlow / Keras (for InceptionV3)
- Scikit-learn (for cosine similarity)
- Pickle (for saving and loading intermediate data)

## Installation

Install the required Python packages using pip:

```bash
pip install pandas numpy tensorflow scikit-learn pickle-mixin
