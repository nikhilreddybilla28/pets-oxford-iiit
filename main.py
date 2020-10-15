#!pip install streamlit

from fastai.vision import *
import streamlit as st
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

#defaults.device = torch.device('cpu')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("PET BREED CLASSIFIER API")

#load model
path_file = Path('.','export.pkl')
learner = load_learner(path_file)

def classifybreed(img):
  pred_class, pred_idx, outputs = learner.predict(img)
  st.write(pred_class)
  return pred_class 

def main():
    html_temp = """
    <div style="background-color:tomato;padding:12px">
    <h2 style="color:white;text-align:center;"> Bredd classifier App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    img_bytes = st.file_uploader("Squash It!!", type=['png', 'jpg', 'jpeg'])
    if img_bytes is not None:
        st.write("Image Uploaded Successfully:")
        img_pil = PIL.Image.open(img_bytes)
        img_tensor = T.ToTensor()(img_pil)
        img = Image(img_tensor)
         

        if st.button("Predict"):
            st.write("Classifying...")
            result = classifybreed(img)
        st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()
