import logging
import time

import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.constant import CLASS_LABELS

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def render(model):
    st.markdown("## Image Classification")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload an image file (PNG, JPG, or JPEG)",
        accept_multiple_files=False,
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file:
        try:
            # Read the image file
            original_image = Image.open(uploaded_file)
            
            ## HNDW 20231023 Start
            # Convert image to RGB if it's not
            img = img.convert("RGB")
            ## HNDW 20231023 End
            
            # Display the original image
            st.image(
                original_image,
                caption="Uploaded Image",
                width=300,
            )

            # Initialize session state for classification
            if "predicted_class" not in st.session_state:
                st.session_state.predicted_class = None

            # Classify Image
            st.markdown("---")
            if st.button("Classify Image"):
                with st.spinner("Classifying image..."):
                    time.sleep(2)
                    # Check for GPU
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    model = model.to(device)

                    # Switch to evaluation mode
                    model.eval()

                    with torch.no_grad():
                        # Preprocess the image
                        img_transform = transforms.Compose(
                            [
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                ),
                            ]
                        )

                        # input_data = (
                        #     img_transform(original_image).unsqueeze(0).to(device)
                        # )

                        # Apply preprocessing
                        img_transform = img_transform(
                            original_image
                        )  # torch.Size([3, 224, 224])

                        input_data = img_transform.unsqueeze(0)
                        # Make predictions
                        predicted = model(input_data)

                        st.write("Input data shape:", input_data.shape)
                        st.write("Predicted scores:", predicted)

                        # Assuming predicted is a tensor of shape (1, num_classes)
                        scores = predicted.cpu().numpy().flatten()
                        df = pd.DataFrame(
                            scores, index=CLASS_LABELS, columns=["Confidence"]
                        )
                        st.bar_chart(df)

                        # Get the predicted class
                        _, pred_label = torch.max(predicted, 1)

                        # Convert index to class name
                        label_name = CLASS_LABELS[pred_label.item()]

                        # Update session state
                        st.session_state.predicted_class = label_name
                        logging.info(f"Predicted class: {label_name}")

            # Display the predicted class
            if st.session_state.predicted_class:
                st.markdown(f"**Predicted Class:** {st.session_state.predicted_class}")

        except Exception as e:
            st.error(f"Error processing the image: {e}")
