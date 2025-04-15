import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL = "apple/aimv2-large-patch14-448"

# main
if __name__ == "__main__":
    #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    file_image = "test.jpg"
    image = Image.open(file_image)

    processor = AutoImageProcessor.from_pretrained(
        MODEL,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        MODEL,
        trust_remote_code=True,
        output_hidden_states=True,
    )

    inputs = processor(images=image, return_tensors="pt")
    print(inputs["pixel_values"].shape)
    outputs = model(**inputs)
    print(outputs["last_hidden_state"][0].shape)
    print(len(outputs["hidden_states"]))
    print(outputs["hidden_states"][0].shape)