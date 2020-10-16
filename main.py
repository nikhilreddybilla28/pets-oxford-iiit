#!pip install streamlit
from fastai.vision import *
import streamlit as st
#from PIL import Image
import urllib.request
import torchvision.transforms as T

#defaults.device = torch.device('cpu')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("PET BREED CLASSIFIER")

introduction_str = 'This is a pet breed classifier.  It was made with 7349 images corresponding to 37 breeds, taken from The Oxford-IIIT Pet Dataset. It was done by transfer of learning using the resnet34 model from Pytorch and the Fastai libraries. \
    In the validation of the model, the training error was 5%.'

st.markdown(introduction_str)

#load model
defaults.device = torch.device('cpu')

MODEL_URL = "https://github.com/nikhilreddybilla28/pets-oxford-iiit/raw/master/export.pkl" 
#"https://drive.google.com/uc?export=download&id=1um_Vk-lilnvOr31D_fSbIbIfGK4sNJoD"
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
path = Path(".")

learner = load_learner(path, "model.pkl")

def classifybreed(img):
  pred_class, pred_idx, outputs = learner.predict(img)
  #st.write(pred_class)
  return pred_class ,pred_idx, outputs


def main():
    html_temp = """
    <div style="background-color:black;padding:10px">
    <h2 style="color:white;text-align:center;"> pet breed classifier </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    img_bytes = st.file_uploader("Squash It!!", type=['png', 'jpg', 'jpeg'])
    if img_bytes is not None:
        st.image(img_bytes, width=150)
        st.write("Image Uploaded Successfully:")
        img = open_image(img_bytes)
        #img_pil = PIL.Image.open(img_bytes)
        #img = pil2fast(img_pil)
        #img = open_image(BytesIO(img_bytes))

        if st.button(""):
            st.write("Classifying...")
            pred_class ,pred_idx, outputs = classifybreed(img)
            out_label = f'Prediction: {pred_class};\n\n Probability: {outputs[pred_idx]:.03f}'
            st.success(out_label)
            #st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()
