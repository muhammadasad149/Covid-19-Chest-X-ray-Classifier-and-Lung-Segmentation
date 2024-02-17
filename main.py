# import streamlit as st
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from keras.models import load_model
# from keras import backend as K
# from PIL import Image

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

# model = load_model('model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

# def process_image(image_path):
#     X_shape = 512
#     x_im = cv2.resize(cv2.imread(image_path), (X_shape, X_shape))[:, :, 0]
#     op = model.predict((x_im.reshape(1, 512, 512, 1) - 127.0) / 127.0)
#     return x_im, op.reshape(512, 512)

# def main():
#     st.title("Image Segmentation App")

#     uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg","png"])
#     if uploaded_file is not None:
#         st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
#         image_path = f"{uploaded_file.name}"
#         with open(image_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         x_im, op = process_image(image_path)

#         st.image(x_im, cmap="bone", label="Input Image", use_column_width=True)
#         st.image(op, cmap="jet", label="Output Image", use_column_width=True)

# if __name__ == "__main__":
#     main()


import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras import backend as K
from PIL import Image
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model = load_model('model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

def process_image(image_path):
    X_shape = 512
    x_im = cv2.resize(cv2.imread(image_path), (X_shape, X_shape))[:, :, 0]
    op = model.predict((x_im.reshape(1, 512, 512, 1) - 127.0) / 127.0)
    return x_im, op.reshape(512, 512)

def main():
    st.title("Image Segmentation App")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg","png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        image_path = f"{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        x_im, op = process_image(image_path)

        # st.image(x_im, caption="Input Image", use_column_width=True)
        
        # Use Matplotlib to display the output image with a specific colormap
        fig, ax = plt.subplots()
        ax.imshow(op, cmap="jet")
        ax.set_title("lungs Segmentation")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
