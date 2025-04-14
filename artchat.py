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


st.title("🎨 AI Art Collaborator")
st.caption(
    """
           Hello! My name is Artie and I want to help you with your art.\n
           I can provide feedback on your artwork, help you brainstorm ideas, and suggest ways to improve your pieces.\n
           You can upload an image of your artwork, and describe what kind of feedback or help you’re looking for.
    """
)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"},
    ]



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


if prompt := st.chat_input():
    # append the user's message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # build the conversation history as input for the model
    conversation_history = system_prompt
    for msg in st.session_state.messages:
        conversation_history += f"\n{msg['role'].capitalize()}: {msg['content']}"

    msg = get_gemini_response(conversation_history)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
