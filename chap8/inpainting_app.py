from PIL import Image, ImageDraw, ImageFilter
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import openai
from io import BytesIO
import urllib.request
from streamlit_cropper import st_cropper

openai.api_key= st.secrets["OPENAI_API_KEY"]

# Do something interesting with the image data and paths
def dall_e(prompt_text,mask,image, size="512x512"):
    response = openai.images.edit(
    image=image,
    mask=mask,
    prompt=prompt_text,
    n=1,
    size=size
    )
    image_url = response.data[0].url
    # urllib.request.urlretrieve(image_url, "image.jpg")
    # img_dalle = Image.open("image.jpg")
    return image_url

def soften_img(image, radius):
    RADIUS = radius
    diam = 2*RADIUS
    back = Image.new('RGB', (image.size[0]+diam, image.size[1]+diam), (232,232,232))
    back.paste(image, (RADIUS, RADIUS))

    # Create paste mask
    mask = Image.new('L', back.size, 0)
    draw = ImageDraw.Draw(mask)
    x0, y0 = 0, 0
    x1, y1 = back.size
    for d in range(diam+RADIUS):
        x1, y1 = x1-1, y1-1
        alpha = 255 if d<RADIUS else int(255*(diam+RADIUS-d)/diam)
        draw.rectangle([x0, y0, x1, y1], outline=alpha)
        x0, y0 = x0+1, y0+1

    # Blur image and paste blurred edge according to mask
    blur = back.filter(ImageFilter.GaussianBlur(RADIUS/2))
    back.paste(blur, mask=mask)
    return back

st.title("PrintAI Demo")
st.markdown("### Welcome to the PrintAI demo")
st.markdown("##### Step 1: Upload an image")
st.markdown("In the sidebar you can upload your image from your local computer. Note: the file should be from the PNG or JPG/JPEG file format")
st.markdown("##### Step 2: Crop the image")
st.markdown("When you have uploaded the image you can use this tool to select the part of the image that you want to use. Note: It is only possible to select a square, if a rectangle is selected the program wont work.")


img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = '#0000FF'

if img_file:
    image_crop = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(image_crop, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=(1, 1))

st.markdown("##### Step 3: inpaint or outpaint the image")
st.markdown("Select inpainting or out painting. In case you choose inpainting draw on the image below to select the area you want to change. In case you choose outpainting draw on the image to select the area you want to keep.")


inpainting = st.radio("Choose inpainting or outpainting",('Inpainting', 'Outpainting'),label_visibility="hidden")

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("rect", "circle", "transform")
)

# stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 3)
stroke_width = 3
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
bg_color = "#eee"
#bg_image = Image.fromarray(np.array(cropped_img))


img = cropped_img
width_img, height_img =img.size


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Black fill color with full opacity
    stroke_width=stroke_width,
    stroke_color="00FFFFFF",
    background_color=bg_color,
    background_image=cropped_img if cropped_img else None,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

st.markdown('##### step 4: Come up with a prompt')
st.markdown('A prompt is a text description of an image. It is important to remember that the prompt should describe the entire image, not just a part of it. For example, if you want to turn yourself into an astronaut, you may use the outpainting tool to select your face. The prompt is what tells the model what to do with that image. A poorly-written prompt might be something like "Add an astronaut helmet and suit" this does not give the model much information to work with, and it only describes a small part of the image. A more effective prompt would be something like "A picture from the 1970s of an astronaut wearing a space suit and helmet, floating in space with planet Mars in the background." This prompt gives the model a lot more information to work with, and it allows the model to produce a more detailed and accurate image.')
prompt_text=st.text_input('The prompt for Dall-e', '')
st.write('The current prompt is:  ', prompt_text)


test = Image.fromarray(canvas_result.image_data)

if inpainting == 'Inpainting':
    new_img = Image.new("RGBA", test.size, (0, 0, 0, 0))

    # Loop over all pixels in the image
    for x in range(test.size[0]):
        for y in range(test.size[1]):
            # Get the RGBA values for the current pixel
            r, g, b, a = test.getpixel((x, y))

            # Invert the alpha value (255 becomes 0, and 0 becomes 255)
            a = 255 - a

            # Set the RGBA values for the current pixel in the new image
            new_img.putpixel((x, y), (r, g, b, a))
    mask = new_img
else:
   mask = Image.fromarray(canvas_result.image_data)


mask = mask.resize((512, 512), Image.BILINEAR)
img = img.resize((512, 512), Image.BILINEAR)

byte_stream = BytesIO()
img.save(byte_stream, format='PNG')
byte_array = byte_stream.getvalue()

byte_stream2 = BytesIO()
mask.save(byte_stream2, format='PNG')
byte_array_mask = byte_stream2.getvalue()


if canvas_result.image_data is not None and cropped_img is not None:
    if st.button('Generate Dall-e image'):
        with st.spinner('Wait for it...'):
            image_url = dall_e(prompt_text,byte_array_mask,byte_array)  
            image = Image.open(urllib.request.urlopen(image_url))
            img_resize = image.resize((245, 252), Image.BILINEAR)

        st.success('Done!')
        st.balloons()
        st.write(image_url)
        st.image(image, caption='Generated Image', use_column_width=True)
        
    else:
        pass
else: 
    pass