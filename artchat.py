import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os

# LLM Libraries
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# See models available in genai
# for m in genai.list_models():
#     print(m.name, m.supported_generation_methods)

# Image Encoding to Base64
import base64
from io import BytesIO


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string):
    buffered = BytesIO(base64.b64decode(base64_string))
    return Image.open(buffered)


# Gemini Helper Functions ###########
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def get_gemini_response_image(input, image):
    model = genai.GenerativeModel("gemini-2.0-flash")
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text


###########################################

st.title("🎨 AI Art Collaborator")
st.caption(
    """
           Hello! My name is Artie and I want to help you with your art.\n
           I can provide feedback on your artwork, help you brainstorm ideas, and suggest ways to improve your pieces.\n
           You can upload an image of your artwork, and describe what kind of feedback or help you’re looking for.
    """
)

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload an image of your art:", type=["jpg", "jpeg", "png"]
    )
    image = ""
    img_input_prompt = st.text_input(
        "Describe what you want feedback on (optional):",
        key="img_input_prompt",
        placeholder="e.g. color palette, composition, etc.",
    )
    image_submitted = st.button("Generate response")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
    ]

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

system_prompt = """
You are an AI art collaborator that provides thoughtful, constructive feedback on visual artworks.
Keep your responses less than 100 words long, and always ask if the user would like more detailed feedback on a specific aspect of their artwork.
The user may upload an image of their artwork, describe what kind of feedback or help they’re looking for, and optionally share a moodboard of images that reflect their intended style or inspiration.
Your role is to analyze the artwork based on the user's goals and give specific, supportive suggestions on visual elements such as color, composition, balance, style alignment, and emotional impact. 
Be encouraging, insightful, and respectful of the artist’s unique voice.
If the user provides a moodboard, consider it when offering feedback to help the user align their piece with their intended aesthetic.
If the user asks for help brainstorming ideas, suggest a few different directions they could take their artwork in, and ask them to elaborate on what resonates with them.
If the user asks for help improving their artwork, ask them to describe what they feel is lacking in their piece and offer specific suggestions on how to enhance it.
If the user asks for help with a specific technique, provide a brief overview of the technique and suggest resources or exercises to practice it.
"""


def get_conversation_history(max_messages=None):
    """
    The conversation history is always rebuilt from st.session_state.messages,
    which ensures that all previous interactions are included
    """
    conversation_history = system_prompt

    if max_messages is not None:
        # Limit the number of messages to include in the conversation history
        max_messages = min(max_messages, len(st.session_state.messages))
        recent_messages = st.session_state.messages[-max_messages:]
    else:
        recent_messages = st.session_state.messages

    for msg in recent_messages:
        conversation_history += f"\n{msg['role'].capitalize()}: {msg['content']}"

    print("Conversation History:")
    print(conversation_history)
    return conversation_history


# React to user input
if prompt := st.chat_input():
    # append the user's message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # generate response based on whether an image is stored
    conversation_history = get_conversation_history()
    if "latest_image" in st.session_state:
        # decode the stored image from base64
        image = decode_base64_to_image(st.session_state["latest_image"])

        msg = get_gemini_response_image(conversation_history, image)
    else:
        msg = get_gemini_response(conversation_history)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

if image_submitted:
    if uploaded_file:
        # display uploaded image on chat window
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

        # encode the image to Base64 and store it in session state
        encoded_image = encode_image_to_base64(image)
        st.session_state["latest_image"] = encoded_image  # Store the latest image

        # append the user's message to the session state
        st.session_state.messages.append(
            {
                "role": "user",
                "content": "Regarding the image, I want feedback on:\n"
                + img_input_prompt,
            }
        )
        st.chat_message("user").write(img_input_prompt)

        # generate response using the image
        conversation_history = get_conversation_history()
        msg = get_gemini_response_image(conversation_history, image)

        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
