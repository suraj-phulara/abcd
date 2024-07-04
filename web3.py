from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Optional
import streamlit as st
import json
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")

# Define the JSON structure for parsing the OpenAI response
class FieldUpdate(BaseModel):
    field_name: str
    answer: str

class ResponseStructure(BaseModel):
    inferences: Optional[List[FieldUpdate]] = None
    next_reply: str
    chat_mode:str


def read_abc_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # print(data)
    return data

# Read the abc.json file
abc_json = read_abc_json('abc.json')



# Sample JSON template (this will be dynamically updated)
tripplan_json = {
    # "optimizeType": {},
    "firstDestination": {},
    "trip_theme": {},
    "destination": {},
    "traveller_type": {},
    "Origin_city": {},
    "budget": {},
    "food": {},
    "trip_direction": {}
}

def call_openai_api(chat_history, current_json):
    if 'function_count' not in st.session_state:
        st.session_state['function_count'] = 0
    else:
        st.session_state['function_count'] += 1
        print("Total number of times the GPT call is made is:", st.session_state['function_count'])

    model = ChatOpenAI(api_key=openai_api_key, temperature=0)

    # Define the parser
    parser = JsonOutputParser(pydantic_object=ResponseStructure)

    example = {
        "optimizeType": "manual",
        "firstDestination": "Goa",
        "trip_theme": "beach",
        "destination": ["Goa", "Kerala"],
        "traveller_type": "family",
        "Origin_city": "Mumbai",
        "budget": "comfortable spending",
        "food": "vegetarian",
        "trip_direction": "return"
    }

    details = {
        "fields": [
            {
            "field_name": "optimizeType",
            "question": "Would you prefer to sequence the list of cities manually, or should we auto-sequence them for you?",
            "options": ["manual", "auto"]
            },
            {
            "field_name": "firstDestination",
            "question": "Where would you like to go first?"
            },
            {
            "field_name": "trip_theme",
            "question": "What is the theme of your trip (e.g., adventure, beach, cultural)?",
            "options": [
                "romantic",
                "family-vacation",
                "eco-tourism",
                "party",
                "roadtrip",
                "remote-work",
                "business-work",
                "health and wellness",
                "spiritual",
                "lbgtq+",
                "adventure",
                "general-tourism-no-theme"
            ]
            },
            {
            "field_name": "destination",
            "question": "Could you provide the list of destinations you plan to visit?",
            "instruction":"this will be a list"
            },
            {
            "field_name": "traveller_type",
            "question": "What type of traveler are you (e.g., solo, family, friends)?",
            "options": [
                "solo",
                "couple",
                "family-no kids",
                "family-with kids",
                "friends"
            ]
            },
            {
            "field_name": "Origin_city",
            "question": "What is your city of origin?"
            },
            {
            "field_name": "budget",
            "question": "How would you describe your budget for this trip?",
            "options": [
                "on a tight budget",
                "comfortable spending",
                "happy to spend for a luxurious vacation"
            ]
            },
            {
            "field_name": "food",
            "question": "Do you have any dietary preferences?",
            "options": [
                "any",
                "Middle-eastern",
                "indian",
                "asian",
                "european",
                "mexican",
                "vegetarian",
                "south american",
                "vegan",
                "seafood",
                "fast food",
                "cafe",
                "dessert",
                "healthy",
                "bar/pub",
                "barbeque",
                "pizza"
            ]
            },
            {
            "field_name": "trip_direction",
            "question": "Is your trip one-way or round-trip?",
            "options": ["return", "oneway"]
            }
        ]
        }



    query = f"""
            You are a travel assistant chatbot. Your name is Travel.AI, and you are designed to help users plan their trips and provide travel-related information. First, you need to get some information from the user. You will receive the current conversation history and a JSON template with the questions that need to be filled based on user inputs.

            You have to interact with the user as a customer service agent and get them to answer questions in order to fill the question JSON template. Ensure your responses are polite, engaging, and context-aware, using natural language to guide the user through the necessary details. Here are some key points to remember:

            1. Engage in a conversational manner to enhance user retention as a customer service agent would do.
            2. Analyze the chat history carefully before responding and giving inferences (chat history contains chat in the ascending order of time of conversation).
            3. each of your reply should be more than 20 words atleast. (most important)
            4. if you think at any place the user is confused or asking for suggestion then change the chat_mode to "suggestion" and and reply with "Sure I can help you decide"
            5. if user refuses to answer any question then or say no to any question or asks to skip then strictly skip that question for now and move on. you bring that question again later in the conversation in a different way(most important)
            6. also allow updation and deletion of answers.
            7. only give the inferences when you have the answer
            8. do not rely on guess work. only give inferences after analyzing the user's answer
            9. if a user tries to skip a question or says no to a question. strictly move to next unanswered question and if it is the only unanswered question left then give a polite reply and explain how important it is to get that data and try asking the same thing in a different tone and way
            10 if to any question the user says he has not decided yet then go to suggestion mode
            11. in the reason always mention your reason behind the inference and if you are not giving inferences then also state why along with latest user input


            Details on each field and how to ask questions: {json.dumps(details, indent=4)}

            Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their travel planning journey.

            Below are a few examples for you to learn how to do it:

            Example 1:

            bot: Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to go for vacation this time?
            User: Paris.
            {{
                "inferences": [
                    {{"field_name": "destination", "answer": "Paris"}}
                ],
                "next_reply": "are you visiting only one city. if there are more than one city you can give me a list of cities you wish to visit",
                "chat_mode": "chat"
            }}

            User: yes I am only going to paris.
            {{
                {{"field_name": "destination", "answer": "Paris"}},
                "next_reply": "Do you have a specific theme for your trip, like cultural, romantic, or maybe an adventure?",
                "chat_mode": "chat"
            }}

            User: I'm planning a wellness and yoga retreat.
            {{
                "inferences": [],
                "next_reply": "It seems like 'wellness and yoga retreat' is not an available option. Could you please choose from the following themes: romantic, family-vacation, eco-tourism, party, roadtrip, remote-work, business-work, health and wellness, spiritual, LGBTQ+, adventure, general-tourism-no-theme?",
                "chat_mode": "chat"
            }}

            User: Health and wellness sounds good.
            {{
                "inferences": [
                    {{"field_name": "trip_theme", "answer": "health and wellness"}}
                ],
                "next_reply": "Could you provide a list of other destinations you plan to visit after Paris?",
                "chat_mode": "chat"
            }}

            User: Sure, we plan to visit Rome and Barcelona as well.
            {{
                "inferences": [
                    {{"field_name": "destination", "answer": ["Paris", "Rome", "Barcelona"]}}
                ],
                "next_reply": "Do you want to manually sequence the cities you will visit, or should we auto-sequence it?",
                "chat_mode": "chat"
            }}

            User: Let's go with the auto-sequencing option.
            {{
                "inferences": [
                    {{"field_name": "optimizeType", "answer": "auto"}}
                ],
                "next_reply": "What is your origin city from where you will leave for vacation?",
                "chat_mode": "chat"
            }}

            User: I am currently living in San Francisco, so it is my origin city.
            {{
                "inferences": [
                    {{"field_name": "Origin_city", "answer": "San Francisco"}}
                ],
                "next_reply": "How would you describe your budget for this trip? Please choose from: 'on a tight budget', 'comfortable spending', 'happy to spend for a luxurious vacation'.",
                "chat_mode": "chat"
            }}

            User: Can we skip that? Let's talk about my food preferences instead.
            {{
            "inferences": [],
            "next_reply": "Sure! Do you have any dietary preferences or restrictions? Please choose from: 'any', 'Middle-eastern', 'indian', 'asian', 'european', 'mexican', 'vegetarian', 'south american', 'vegan', 'seafood', 'fast food', 'cafe', 'dessert', 'healthy', 'bar/pub', 'barbeque', 'pizza'.",
                "chat_mode": "chat"
            }}

            User: I prefer Asian cuisine.
            {{
            "inferences": [
                {{"field_name": "food", "answer": "asian"}}
            ],
            "next_reply": "Got it! Now, it would really help me to know your budget for this trip to give you the best recommendations. Can you choose from: 'on a tight budget', 'comfortable spending', 'happy to spend for a luxurious vacation'?",
                "chat_mode": "chat"
            }}

            User: I'd rather not say.
            {{
            "inferences": [],
            "next_reply": "No worries! Let's try this: Are you looking for budget-friendly options, mid-range comforts, or luxury experiences?",
                "chat_mode": "chat",
            "reason": "the user is trying to skip the question so i am not giving any inferences as the user has not answered anything"
            }}

            User: Mid-range comforts.
            {{
            "inferences": [
                {{"field_name": "budget", "answer": "comfortable spending"}}
            ],
            "next_reply": "Great! Is your trip going to be one-way or a round trip? Please choose from: 'return', 'oneway'.",
                "chat_mode": "chat",
            "reason": "the user mentioned mid range comforts in the latest response so i am choosing that as the answer"
            }}



            Example 2:

            Bot: Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to go for vacation this time?
            User: Hi, I am not sure where to start. Can you suggest some popular destinations?
            {{
                "inferences": [],
                "next_reply": "Sure! Could you tell me a bit about your preferences? Are you looking for cultural experiences, adventure, relaxation, or something else? also i am changing the mode to sggestion",
                "chat_mode": "suggestion",
            "reason": "the user is asking for suggestion"
            }}



            Example 3:

            Bot: Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to visit this time for vacation?
            User: You suggest me some.
            {{
            "inferences": [],
            "next_reply": "Sure I can help you decide! Could you tell me a bit about your preferences? Are you looking for cultural experiences, adventure, relaxation, or something else? also i am changing the mode to sggestion",
            "chat_mode": "suggestion",
            "reason": "the user is asking for suggestion"
            }}

            Remember the above are just some examples for you to learn from. Do not include this data in your inferences; base your output on the conversation in chat history.

            
            know that the above examples contain the entire chat whereas you have to only take the recent chat history and give inference and the next reply or question only in your answer and nothing else.

            your output json should have this structure :

            {{
                "inferences": [
                    
                ],
                "next_reply": "",
                "chat_mode": "",
                "reason": "here you have to give your reasons for whatever inference you gave along with latest user input. example - i am giving inference for destination to be tokyo as the user mentioned tokyo in this latest input ie. 'i wish to to to tokyo' and if you are leaving the inference empty then state why so"
            }}

            just give me this json as output nothing else
            your inferences should be based on latest chat history nd relevent to what is going on here : {chat_history[-5:-1]}
            if the suggestion is going on then give suggestion only and help the user decide. base your suggestion on the questions already answered.

            also you can give inferences even for a one word answer based on the context from the chat_hist0ry

            
            Chat history:
            {chat_history}

            Questions that need to be answered (some of them have been answered; focus on those that are still not answered):
            {json.dumps(current_json, indent=4)}

            remember to chek if before giving inferences if the user has answered or not. see the latest user input in chat history if you want
            also if you cannot match the answer to the value options then ask the user to clarify 

            only when all the questions have been answered say: "thankyou i have all the info i need"

            also you can ask questions in any order you would like

            also do not use any made up data or the data from the examples they were just for you to learn.

            and finally the most important thing: before asking any question analyze the entire chat history for if the question is already answered or not
            

            """


    # Define the prompt template
    prompt = PromptTemplate(
        template="""Answer the user query.\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the LLMChain
    chain = prompt | model | parser

    # Prepare the input for the chain
    input_data = {
        "query":query
    }

    # Invoke the chain
    response = chain.invoke({"query":query})

    # Convert response to dictionary
    updated_json = response

    # Update the current JSON with the new answers
    if "inferences" in updated_json and updated_json["inferences"]:
        for update in updated_json["inferences"]:
            current_json[update["field_name"]] = update["answer"]

    print(chat_history[-1],"     response:", json.dumps(response, indent=4))

    return current_json, updated_json.get("next_reply", "")

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        chatbot_intro = "Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to visit this time for vacation"
        st.session_state['chat_history'].append(f"Bot: {chatbot_intro}")
    if 'json_data' not in st.session_state:
        st.session_state['json_data'] = tripplan_json
    if 'next_reply' not in st.session_state:
        st.session_state['next_reply'] = ""
    if 'function_count' not in st.session_state:
        st.session_state['function_count'] = 0

def handle_user_input(user_input):
    """
    Handle user input: update chat history, call OpenAI API, and update JSON data.
    
    Args:
        user_input (str): The user input text.
    """
    st.session_state['chat_history'].append(f"User: {user_input}")

    # Call OpenAI API to update JSON data
    updated_json, next_reply = call_openai_api(
        st.session_state['chat_history'],
        st.session_state['json_data']
    )
    st.session_state['json_data'] = updated_json
    st.session_state['next_reply'] = next_reply

    # Generate a chatbot response
    chatbot_response = f"{next_reply}"
    st.session_state['chat_history'].append(f"Bot: {chatbot_response}")

def render_chatbot_ui():
    """Render the chatbot user interface."""
    st.header("Chatbot")
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    user_input = st.text_input("User:", "")
    
    if user_input:
        print("user input is:", user_input)
        handle_user_input(user_input)
    
    chat_history_container = st.container()
    with chat_history_container:
        for chat in st.session_state['chat_history']:
            st.write(chat)

def render_json_ui():
    """Render the JSON data user interface."""
    st.header("Current JSON Data")
    st.json(st.session_state['json_data'])

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Travel Planner Chatbot")

    initialize_session_state()

    # Create two columns with wider width
    col1, col2 = st.columns([2, 2])

    # Left column: Chatbot interaction
    with col1:
        render_chatbot_ui()

    # Right column: Display JSON data
    with col2:
        render_json_ui()


def clear_session_state():
    """Clear all session state variables."""
    for key in st.session_state.keys():
        del st.session_state[key]

if __name__ == "__main__":
    main()
