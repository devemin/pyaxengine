# Copyright (c) 2024 nnn112358
# modified 2024/12/08 by devemin


############################################
#
# Edit in line 130 and line 178
#
############################################


import axengine as axe
import numpy as np
import cv2
import time 
from PIL import Image
import copy

def load_model(model_path):
    session = axe.InferenceSession(model_path)

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print("\nModel Information:")
    print("Inputs:")
    for input in inputs:
        print(f"- Name: {input.name}")
        print(f"- Shape: {input.shape}")
#        print(f"- Type: {input.type}")
    print("\nOutputs:")
    for output in outputs:
        print(f"- Name: {output.name}")
        print(f"- Shape: {output.shape}")
#        print(f"- Type: {output.type}")

    return session




#https://qiita.com/derodero24/items/f22c22b22451609908ee
#https://imagingsolution.net/program/python/numpy/python_numpy_pillow_image_convert/
def cv2pil(image):
    ''' OpenCV -> PIL '''
    new_image = image.copy()
    if new_image.ndim == 2:  # Mono
        pass
    elif new_image.shape[2] == 3:  # Color
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # Transparent
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
    


def preprocess_image(img_data, target_size=(256, 256), crop_size=(224, 224)):

    img = copy.deepcopy(img_data)

    # Get original dimensions
    original_width, original_height = img.size

    # Determine the shorter side and calculate the center crop
    if original_width < original_height:
        crop_area = original_width
    else:
        crop_area = original_height

    crop_x = (original_width - crop_area) // 2
    crop_y = (original_height - crop_area) // 2

    # Crop the center square
    img = img.crop((crop_x, crop_y, crop_x + crop_area, crop_y + crop_area))

    # Resize the image to 256x256
    img = img.resize(target_size)

    # Crop the center 224x224
    crop_x = (target_size[0] - crop_size[0]) // 2
    crop_y = (target_size[1] - crop_size[1]) // 2
    img = img.crop((crop_x, crop_y, crop_x + crop_size[0], crop_y + crop_size[1]))

    # Convert to numpy array and change dtype to int
    img_array = np.array(img).astype("uint8")
    # Transpose to (1, C, H, W)
    # img_array = np.transpose(img_array, (2, 0, 1))
    # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def get_top_k_predictions(output, k=5):
    print("\nOutput tensor information:")
    print(f"Shape: {output[0].shape}")
    print(f"dtype: {output[0].dtype}")
    print(f"Value range: [{output[0].min():.3f}, {output[0].max():.3f}]")

    # Get top k predictions
    top_k_indices = np.argsort(output[0].flatten())[-k:][::-1]
    top_k_scores = output[0].flatten()[top_k_indices]
    return top_k_indices, top_k_scores






def main(model_path, image_path, target_size, crop_size, k):
    WIDTH = 320
    HEIGHT = 240

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    #cap.set(cv2.CAP_PROP_FPS, 10)


    # Load the model
    session = load_model(model_path)


    #for i in range(100):
    while(True):
        ret, frame = cap.read()
        #print("#############################")
        #print(ret)
    
        # Load the image
        
        ### Change the Cat or Camera ### 
        CAT_OR_CAM = 0
        
        if (CAT_OR_CAM):
            #input from cat.jpg
            img_data = Image.open(image_path).convert("RGB")
        else:
            #input from camera
            pil_image = cv2pil(frame)
            #numpy_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #pil_image = Image.fromarray(frame)
            img_data = pil_image.convert("RGB")
            #pass
        
        
        # Preprocess the image
        input_tensor = preprocess_image(img_data, target_size, crop_size)
    
        print(input_tensor.shape)
    
        # Get input name and run inference
        input_name = session.get_inputs()[0].name
    
        print("Input name:", input_name)
        print("Input tensor shape:", input_tensor.shape)
        print("Expected input shape:", session.get_inputs()[0].shape)
        print("Expected output shape:", session.get_outputs()[0].shape)
    
        #print(f"\nIteration {i+1}/100")
        
        # Measure inference time
        start_time = time.time()
        output = session.run(None, {input_name: input_tensor})
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"Inference Time: {inference_time:.2f} ms")
        
        # Get top k predictions
        top_k_indices, top_k_scores = get_top_k_predictions(output, k)
        
        # Print results for this iteration
        print(f"Top {k} Predictions:")
        for j in range(k):
            print(f"Class Index: {top_k_indices[j]}, Score: {top_k_scores[j]}")

        #time.sleep(5)

        ### Change the comment-out, but now X11Forward not working. ### 
        #cv2.imshow('camera' , frame)



if __name__ == "__main__":
    MODEL_PATH = "/opt/data/npu/models/mobilenetv2.axmodel"
    IMAGE_PATH = "/opt/data/npu/images/cat.jpg"
    TARGET_SIZE = (256, 256)  # Resize to 256x256
    CROP_SIZE = (224, 224)  # Crop to 224x224
    K = 5  # Top K predictions
    main(MODEL_PATH, IMAGE_PATH, TARGET_SIZE, CROP_SIZE, K)






