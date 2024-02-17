# Covid-19 Chest X-ray Classifier and Lung Segmentation

This Streamlit-based web application classifies chest X-ray images into COVID-19, Normal, or Viral pneumonia categories using various pre-trained deep learning models (VGG16, VGG19, ResNet50, ResNet101, Inception). Additionally, it performs lung segmentation on the uploaded images.

## Features

- Chest X-ray image classification into COVID-19, Normal, or Viral pneumonia categories.
- Lung segmentation visualization.
- Selection of different pre-trained classification models.
- Real-time inference on uploaded images.
- Easy-to-use interface powered by Streamlit.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/muhammadasad149/Covid-19-Chest-X-ray-Classifier-and-Lung-Segmentation.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Select the desired classification model from the dropdown menu.
3. Upload a chest X-ray image.
4. View the classification prediction and lung segmentation result.

## File Structure

- `app.py`: Main Streamlit application script.
- [Download the pre-trained segmentation model (model.h5)]([https://drive.google.com/your_model_link](https://drive.google.com/drive/folders/1C4WDeYMF98ijY81uRQYLOz0tBOhzLBuF?usp=sharing))
- [Download the pre-trained VGG16 classification model (VGG16_model.h5)]([https://drive.google.com/your_model_link](https://drive.google.com/drive/folders/1C4WDeYMF98ijY81uRQYLOz0tBOhzLBuF?usp=sharing))
- [Download the pre-trained VGG19 classification model (VGG19_model.h5)]([https://drive.google.com/your_model_link](https://drive.google.com/drive/folders/1C4WDeYMF98ijY81uRQYLOz0tBOhzLBuF?usp=sharing))
- [Download the pre-trained ResNet50 classification model (ResNet50_model.h5)]([https://drive.google.com/your_model_link](https://drive.google.com/drive/folders/1C4WDeYMF98ijY81uRQYLOz0tBOhzLBuF?usp=sharing))
- [Download the pre-trained ResNet101 classification model (ResNet101_model.h5)]([https://drive.google.com/your_model_link](https://drive.google.com/drive/folders/1C4WDeYMF98ijY81uRQYLOz0tBOhzLBuF?usp=sharing))
- [Download the pre-trained Inception classification model (Inception_model.h5)]([https://drive.google.com/your_model_link](https://drive.google.com/drive/folders/1C4WDeYMF98ijY81uRQYLOz0tBOhzLBuF?usp=sharing))
- `requirements.txt`: List of Python dependencies.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request.
