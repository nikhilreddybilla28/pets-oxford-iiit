#!pip install streamlit
from fastai.vision import *
import streamlit as st
#from PIL import Image
import urllib.request
import torchvision.transforms as T

#defaults.device = torch.device('cpu')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("PET BREED CLASSIFIER")

#load model
defaults.device = torch.device('cpu')

MODEL_URL = "https://github.com/nikhilreddybilla28/pets-oxford-iiit/raw/master/export.pkl" 
#"https://drive.google.com/uc?export=download&id=1um_Vk-lilnvOr31D_fSbIbIfGK4sNJoD"
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
path = Path(".")

learner = load_learner(path, "model.pkl")

def classifybreed(img):
  pred_class, pred_idx, outputs = learner.predict(img)
  st.write(pred_class)
  return pred_class 

def pil2fast(img):
  return Image(T.ToTensor()(img))

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
        img = open_image(img_bytes)
        #img_pil = PIL.Image.open(img_bytes)
        #img = pil2fast(img_pil)
        #img = open_image(BytesIO(img_bytes))

        if st.button("Predict"):
            st.write("Classifying...")
            result = classifybreed(img)
            st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()
