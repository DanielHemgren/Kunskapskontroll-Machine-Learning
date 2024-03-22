import streamlit as st
from skimage import io, color, transform, util
import joblib
from skimage import data
from skimage.color import rgb2gray




st.title("Daniel Hemgrens image recognition")

uploaded_file = st.file_uploader("Choose a digit image", type=["jpg"])

# kontroll av filuppladdning
if uploaded_file:
    img_path = 'tmpfile.jpg'
    with open(img_path, 'wb') as f:
        f.write(uploaded_file.read())  # Save the file

    # Bildtransformer
    mnist7_img = io.imread(img_path)
    st.image(mnist7_img, caption='Uploaded Image', use_column_width=True)


    mnist7_2828 = transform.resize(mnist7_img, (28, 28))

   
    mnist7_gray = color.rgb2gray(mnist7_2828)

    mnist7_gray2 = mnist7_gray * 255

   
    threshold_value = 0.8
    background_mask = mnist7_gray < threshold_value

    mnist7_white = util.img_as_ubyte(mnist7_gray.copy())

   
    mnist7_flat = mnist7_white.flatten()

   
    inv = 255 - mnist7_flat

    # modell laddas
    voting_clf = joblib.load('voting_clf3.joblib')

    # Prediktion
    pred_digit = voting_clf.predict([inv])

    print("Siffran är!:", pred_digit[0])
    st.text(f"The predicted digit is: {pred_digit[0]}")