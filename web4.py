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
            # ,
            # {
            # "field_name": "time_schedule",
            # "question": "Could you provide details for your trip schedule?",
            # "sub_fields": [
            #     {
            #     "field_name": "onward_trip",
            #     "question": "When would you like to start your trip?",
            #     "sub_fields": [
            #         {
            #         "field_name": "time",
            #         "question": "At what time would you like to depart?",
            #         "sub_fields": [
            #             {
            #             "field_name": "hour",
            #             "question": "Hour:"
            #             },
            #             {
            #             "field_name": "minute",
            #             "question": "Minute:"
            #             },
            #             {
            #             "field_name": "format",
            #             "question": "Time format:",
            #             "options": ["AM/PM", "24Hour"]
            #             }
            #         ]
            #         },
            #         {
            #         "field_name": "date",
            #         "question": "What date would you like to depart?",
            #         "sub_fields": [
            #             {
            #             "field_name": "day_of_month",
            #             "question": "Day of the month:",
            #             "options": [1, 31]
            #             },
            #             {
            #             "field_name": "month",
            #             "question": "Month:",
            #             "options": [1, 12]
            #             },
            #             {
            #             "field_name": "year",
            #             "question": "Year:",
            #             "options": ["current_year", "next_year"]
            #             }
            #         ]
            #         }
            #     ]
            #     },
            #     {
            #     "field_name": "return_trip",
            #     "question": "When would you like to return?",
            #     "sub_fields": [
            #         {
            #         "field_name": "time",
            #         "question": "At what time would you like to depart?",
            #         "sub_fields": [
            #             {
            #             "field_name": "hour",
            #             "question": "Hour:"
            #             },
            #             {
            #             "field_name": "minute",
            #             "question": "Minute:"
            #             },
            #             {
            #             "field_name": "format",
            #             "question": "Time format:",
            #             "options": ["AM/PM", "24Hour"]
            #             }
            #         ]
            #         },
            #         {
            #         "field_name": "date",
            #         "question": "What date would you like to return?",
            #         "sub_fields": [
            #             {
            #             "field_name": "day_of_month",
            #             "question": "Day of the month:",
            #             "options": [1, 31]
            #             },
            #             {
            #             "field_name": "month",
            #             "question": "Month:",
            #             "options": [1, 12]
            #             },
            #             {
            #             "field_name": "year",
            #             "question": "Year:",
            #             "options": ["current_year", "next_year"]
            #             }
            #         ]
            #         }
            #     ]
            #     },
            #     {
            #     "field_name": "duration",
            #     "question": "What is the duration of your trip?",
            #     "sub_fields": [
            #         {
            #         "field_name": "value",
            #         "question": "Duration value:"
            #         },
            #         {
            #         "field_name": "unit",
            #         "question": "Duration unit:",
            #         "options": ["week", "month", "days"]
            #         }
            #     ]
            #     }
            # ]
            # }
        ]
        }


    # query1 = f"""

    #     You are a travel assistant chatbot your name is Travel.AI designed to help users plan their trips and provide travel-related information. But First you need to get some infromation from the user. You will receive the current conversation history and a JSON template having the questions that needs to be filled based on user inputs. 
            
    #         You have to interact with the user as a customer service agent and get them to answer questions in order to fill the question JSON template. Ensure your responses are polite, engaging, and context-aware, using natural language to guide the user through the necessary details. Here are some key points to remember:

    #             1. Engage in a conversational manner to enhance user retention.
    #             2. Analyze the chat history carefully before responding.
    #             3. Avoid repeating questions that have already been answered.
    #             4. Tailor your questions based on the context and previous interactions to make the conversation flow naturally.
    #             5. If the user asks for suggestion then provide proper suggestions and guide thim till the user has made a decision. use chat history to deside if the conversation is in suggestion state or not
    #             6. if the user asks for suggestions Provide information about popular travel destinations. Offer insights on attractions, local cuisine, cultural highlights, weather conditions, and best times to visit, Suggest local attractions and points of interest based on the user's destination. Highlight must-see landmarks, museums, parks, and recreational activities.
    #             7. Offer practical travel tips and advice. Topics may include packing essentials, visa requirements, currency exchange, local customs, and safety tips.
    #             8. Address customer service inquiries and provide assistance with travel-related issues.
                

            
    #             Details on each field and how to ask questions: {json.dumps(details, indent=4)}
            
    #         Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their travel planning journey.

    #         below are few examples for you to learn how to do it:


    #             example 1-

    #                 bot : Hello! I am here to help you plan your vacation. Let's get started! What is are you planning to go for vacation this time.?
    #                 User: Hi, We are thinking about Paris. 
    #                 {
    #                     "inferences": [
    #                         {"field_name": "firstDestination", "answer": "Paris"}
    #                     ],
    #                     "next_reply": "Do you have a specific theme for your trip, like cultural, romantic, or maybe an adventure?"
    #                 }

    #                 User: I'm planning a wellness and yoga retreat.
    #                 {
    #                     "inferences": [],
    #                     "next_reply": "It seems like 'wellness and yoga retreat' is not an available option. Could you please choose from the following themes: romantic, family-vacation, eco-tourism, party, roadtrip, remote-work, business-work, health and wellness, spiritual, LGBTQ+, adventure, general-tourism-no-theme?"
    #                 }

    #                 User: Health and wellness sounds good.
    #                 {
    #                     "inferences": [
    #                         {"field_name": "trip_theme", "answer": "health and wellness"}
    #                     ],
    #                     "next_reply": "Could you provide a list of other destinations you plan to visit after Paris?"
    #                 }

    #                 User: Sure, we plan to visit Rome and Barcelona as well.
    #                 {
    #                     "inferences": [
    #                         {"field_name": "destination", "answer": ["Paris", "Rome", "Barcelona"]}
    #                     ],
    #                     "next_reply": "next_reply": "Do you want to manually sequence the cities you will visit, or should we auto-sequence it?"
    #                 }

    #                 User: Let's go with the auto-sequencing option.
    #                 {
    #                     "inferences": [
    #                         {"field_name": "optimizeType", "answer": "auto"}
    #                     ],
    #                     "next_reply": "What is your origin city from where you will leave for vacation"
    #                 }

    #                 User: I am currently living in San Francisco so it is my origin city.
    #                 {
    #                     "inferences": [
    #                         {"field_name": "Origin_city", "answer": "San Francisco"}
    #                     ],
    #                     "next_reply": "How would you describe your budget for this trip? Please choose from: 'on a tight budget', 'comfortable spending', 'happy to spend for a luxurious vacation'."
    #                 }

    #                 User: Can we skip that? Let's talk about my food preferences instead.
    #                 {
    #                 "inferences": [],
    #                 "next_reply": "Sure! Do you have any dietary preferences or restrictions? Please choose from: 'any', 'Middle-eastern', 'indian', 'asian', 'european', 'mexican', 'vegetarian', 'south american', 'vegan', 'seafood', 'fast food', 'cafe', 'dessert', 'healthy', 'bar/pub', 'barbeque', 'pizza'."
    #                 }

    #                 User: I prefer Asian cuisine.
    #                 {
    #                 "inferences": [
    #                     {"field_name": "food", "answer": "asian"}
    #                 ],
    #                 "next_reply": "Got it! Now, it would really help me to know your budget for this trip to give you the best recommendations. Can you choose from: 'on a tight budget', 'comfortable spending', 'happy to spend for a luxurious vacation'?"
    #                 }

    #                 User: I'd rather not say.
    #                 {
    #                 "inferences": [],
    #                 "next_reply": "No worries! Let's try this: Are you looking for budget-friendly options, mid-range comforts, or luxury experiences?"
    #                 }

    #                 User: Mid-range comforts.
    #                 {
    #                 "inferences": [
    #                     {"field_name": "budget", "answer": "comfortable spending"}
    #                 ],
    #                 "next_reply": "Great! Is your trip going to be one-way or a round trip? Please choose from: 'return', 'oneway'."
    #                 }


    #             example 2-

    #             Bot: Hello! I am here to help you plan your vacation. Let's get started! What is are you planning to go for vacation this time.?
    #             User: Hi, I am not sure where to start. Can you suggest some popular destinations?
    #             {
    #                 "inferences": [],
    #                 "next_reply": "Sure! Could you tell me a bit about your preferences? Are you looking for cultural experiences, adventure, relaxation, or something else?"
    #             }

    #             User: I'm looking for a mix of cultural experiences and adventure.
    #             {
    #             "inferences": [
    #                 {"field_name": "trip_theme", "answer": "cultural and adventure"}
    #             ],
    #             "next_reply": "Great! Based on that, some popular destinations are Tokyo, Kyoto, Rome, and Barcelona. Which one appeals to you the most?"
    #             }

    #             User: Let's start with Tokyo.

    #             and soo on



    #         remember the above are just some examples for you to learn from do not include this data in your inferences base your output on the conversation in chat history

            

    #         Chat history:
    #         {chat_history}

    #         question that need to be answered (some of them have been answered focus on those that are still not answered):
    #         {json.dumps(current_json, indent=4)}
    
    # """

        # 12. if at any place you think user is asking for suggestions then just reply with : "you can ask my friend suraj for any suggestion he is best at it I am handling control over to him"
    # query2 = f"""
    #     You are a travel assistant chatbot. Your name is Travel.AI, and you are designed to help users plan their trips and provide travel-related information. First, you need to get some information from the user. You will receive the current conversation history and a JSON template with the questions that need to be filled based on user inputs.

    #     You have to interact with the user as a customer service agent and get them to answer questions in order to fill the question JSON template. Ensure your responses are polite, engaging, and context-aware, using natural language to guide the user through the necessary details. Here are some key points to remember:

    #     1. Engage in a conversational manner to enhance user retention.
    #     2. Analyze the chat history carefully before responding.
    #     3. Avoid repeating questions that have already been answered.
    #     4. Tailor your questions based on the context and previous interactions to make the conversation flow naturally.
    #     5. If the user asks for suggestions, provide proper suggestions and guide them until they have made a decision. Use chat history to determine if the conversation is in suggestion state or not.
    #     6. If the user asks for suggestions, provide information about popular travel destinations. Offer insights on attractions, local cuisine, cultural highlights, weather conditions, and best times to visit. Suggest local attractions and points of interest based on the user's destination. Highlight must-see landmarks, museums, parks, and recreational activities.
    #     7. Offer practical travel tips and advice. Topics may include packing essentials, visa requirements, currency exchange, local customs, and safety tips.
    #     8. Address customer service inquiries and provide assistance with travel-related issues.
    #     9. Remember a good customer service never asks the same question twice so slightly change the questions every time you need to ask same question twice.
    #     10. each of your reply should be more than 20 words atleast. (most important)
    #     11. if there is any confusion you have you can ask the user directly for clarity


    #     Details on each field and how to ask questions: {json.dumps(details, indent=4)}

    #     Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their travel planning journey.

    #     Below are a few examples for you to learn how to do it:

    #     Example 1:

    #     bot: Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to go for vacation this time?
    #     User: Hi, We are thinking about Paris.
    #     {{
    #         "inferences": [
    #             {{"field_name": "firstDestination", "answer": "Paris"}}
    #         ],
    #         "next_reply": "Do you have a specific theme for your trip, like cultural, romantic, or maybe an adventure?",
    #         "chat_mode": "chat"
    #     }}

    #     User: I'm planning a wellness and yoga retreat.
    #     {{
    #         "inferences": [],
    #         "next_reply": "It seems like 'wellness and yoga retreat' is not an available option. Could you please choose from the following themes: romantic, family-vacation, eco-tourism, party, roadtrip, remote-work, business-work, health and wellness, spiritual, LGBTQ+, adventure, general-tourism-no-theme?",
    #         "chat_mode": "chat"
    #     }}

    #     User: Health and wellness sounds good.
    #     {{
    #         "inferences": [
    #             {{"field_name": "trip_theme", "answer": "health and wellness"}}
    #         ],
    #         "next_reply": "Could you provide a list of other destinations you plan to visit after Paris?",
    #         "chat_mode": "chat"
    #     }}

    #     User: Sure, we plan to visit Rome and Barcelona as well.
    #     {{
    #         "inferences": [
    #             {{"field_name": "destination", "answer": ["Paris", "Rome", "Barcelona"]}}
    #         ],
    #         "next_reply": "Do you want to manually sequence the cities you will visit, or should we auto-sequence it?",
    #         "chat_mode": "chat"
    #     }}

    #     User: Let's go with the auto-sequencing option.
    #     {{
    #         "inferences": [
    #             {{"field_name": "optimizeType", "answer": "auto"}}
    #         ],
    #         "next_reply": "What is your origin city from where you will leave for vacation?",
    #         "chat_mode": "chat"
    #     }}

    #     User: I am currently living in San Francisco, so it is my origin city.
    #     {{
    #         "inferences": [
    #             {{"field_name": "Origin_city", "answer": "San Francisco"}}
    #         ],
    #         "next_reply": "How would you describe your budget for this trip? Please choose from: 'on a tight budget', 'comfortable spending', 'happy to spend for a luxurious vacation'.",
    #         "chat_mode": "chat"
    #     }}

    #     User: Can we skip that? Let's talk about my food preferences instead.
    #     {{
    #     "inferences": [],
    #     "next_reply": "Sure! Do you have any dietary preferences or restrictions? Please choose from: 'any', 'Middle-eastern', 'indian', 'asian', 'european', 'mexican', 'vegetarian', 'south american', 'vegan', 'seafood', 'fast food', 'cafe', 'dessert', 'healthy', 'bar/pub', 'barbeque', 'pizza'.",
    #         "chat_mode": "chat"
    #     }}

    #     User: I prefer Asian cuisine.
    #     {{
    #     "inferences": [
    #         {{"field_name": "food", "answer": "asian"}}
    #     ],
    #     "next_reply": "Got it! Now, it would really help me to know your budget for this trip to give you the best recommendations. Can you choose from: 'on a tight budget', 'comfortable spending', 'happy to spend for a luxurious vacation'?",
    #         "chat_mode": "chat"
    #     }}

    #     User: I'd rather not say.
    #     {{
    #     "inferences": [],
    #     "next_reply": "No worries! Let's try this: Are you looking for budget-friendly options, mid-range comforts, or luxury experiences?",
    #         "chat_mode": "chat"
    #     }}

    #     User: Mid-range comforts.
    #     {{
    #     "inferences": [
    #         {{"field_name": "budget", "answer": "comfortable spending"}}
    #     ],
    #     "next_reply": "Great! Is your trip going to be one-way or a round trip? Please choose from: 'return', 'oneway'.",
    #         "chat_mode": "chat"
    #     }}

    #     Example 2:

    #     Bot: Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to go for vacation this time?
    #     User: Hi, I am not sure where to start. Can you suggest some popular destinations?
    #     {{
    #         "inferences": [],
    #         "next_reply": "Sure! Could you tell me a bit about your preferences? Are you looking for cultural experiences, adventure, relaxation, or something else?",
    #         "chat_mode": "suggestion"
    #     }}

    #     User: I'm looking for a mix of cultural experiences and adventure.
    #     {{
    #     "inferences": [
    #         {{"field_name": "trip_theme", "answer": "cultural and adventure"}}
    #     ],
    #     "next_reply": "Great! Based on that, some popular destinations are Tokyo, Kyoto, Rome, and Barcelona. Which one appeals to you the most?",
    #         "chat_mode": "suggestion"
    #     }}

    #     User: Let's start with Tokyo.
    #     {{
    #     "inferences": [
    #         {{"field_name": "trip_theme", "answer": "cultural and adventure"}}
    #     ],
    #     "next_reply": "Great! Based on that, some popular destinations are Tokyo, Kyoto, Rome, and Barcelona. Which one appeals to you the most?",
    #         "chat_mode": "suggestion"
    #     }}

    #     And so on.

    #     Remember the above are just some examples for you to learn from. Do not include this data in your inferences; base your output on the conversation in chat history.

    #     know that the above examples contain the entire chat whereas you have to only take the recent chat history and give inference and the next reply or question only in your answer and nothing else.

    #     your output json should have this structure :

    #     {{
    #         "inferences": [
                
    #         ],
    #         "next_reply": "",
    #         "chat_mode": ""
    #     }}

    #     just give me this json as output nothing else
    #     your inferences should be based on latest chat history nd relevent to what is going on here : {chat_history[-5:-1]}
    #     if the suggestion is going on then give suggestion only and help the user decide



    #     Chat history:
    #     {chat_history}

    #     Questions that need to be answered (some of them have been answered; focus on those that are still not answered):
    #     {json.dumps(current_json, indent=4)}
    #     """

    # print(query)





    # example_temp = """

    #     You are a travel assistant chatbot your name is Travel.AI designed to help users plan their trips and provide travel-related information. But First you need to get some infromation from the user. You will receive the current conversation history and a JSON template having the questions that needs to be filled based on user inputs. 
        
    #     You have to interact with the user as a customer service agent and get them to answer questions in order to fill the question JSON template. Ensure your responses are polite, engaging, and context-aware, using natural language to guide the user through the necessary details. Here are some key points to remember:

    #         1. Engage in a conversational manner to enhance user retention.
    #         2. Analyze the chat history carefully before responding.
    #         3. Avoid repeating questions that have already been answered.
    #         4. Tailor your questions based on the context and previous interactions to make the conversation flow naturally.
    #         5. If the user asks for suggestion then provide proper suggestions and guide thim till the user has made a decision. use chat history to deside if the conversation is in suggestion state or not
    #         6. if the user asks for suggestions Provide information about popular travel destinations. Offer insights on attractions, local cuisine, cultural highlights, weather conditions, and best times to visit, Suggest local attractions and points of interest based on the user's destination. Highlight must-see landmarks, museums, parks, and recreational activities.
    #         7. Offer practical travel tips and advice. Topics may include packing essentials, visa requirements, currency exchange, local customs, and safety tips.
    #         8. Address customer service inquiries and provide assistance with travel-related issues.

        
    #     Details on each field and how to ask questions:
        
        
    #     Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their travel planning journey.

    #     Chat history:
    #     {chat_history}

    #     question that need to be answered (some of them have been answered focus on those that are still not answered):
    #     {current_json}
                    
        
        
    #     Here’s how you should handle the scenario:"

    #     1. Travel Tips: Offer practical travel tips and advice. Topics may include packing essentials, visa requirements, currency exchange, local customs, and safety tips.


    #     2. Customer Service: Address customer service inquiries and provide assistance with travel-related issues. Handle queries about bookings, cancellations, refunds, and general support.




    #     You are a  travel assistant chatbot your name is Travel.AI designed to help users plan their trips and provide travel-related information.

    #     Your main task is to interact with the user and trying to get info about  
        
        
        
        
        
    #     Here are some scenarios you should be able to handle:

    #     1. Booking Flights: Assist users with booking flights to their desired destinations. Ask for departure city, destination city, travel dates, and any specific preferences (e.g., direct flights, airline preferences). Check available airlines and book the tickets accordingly.

    #     2. Booking Hotels: Help users find and book accommodations. Inquire about city or region, check-in/check-out dates, number of guests, and accommodation preferences (e.g., budget, amenities). 

    #     3. Booking Rental Cars: Facilitate the booking of rental cars for travel convenience. Gather details such as pickup/drop-off locations, dates, car preferences (e.g., size, type), and any additional requirements.

    #     4. Destination Information: Provide information about popular travel destinations. Offer insights on attractions, local cuisine, cultural highlights, weather conditions, and best times to visit.

    #     5. Travel Tips: Offer practical travel tips and advice. Topics may include packing essentials, visa requirements, currency exchange, local customs, and safety tips.

    #     6. Weather Updates: Give current weather updates for specific destinations or regions. Include temperature forecasts, precipitation chances, and any weather advisories.

    #     7. Local Attractions: Suggest local attractions and points of interest based on the user's destination. Highlight must-see landmarks, museums, parks, and recreational activities.

    #     8. Customer Service: Address customer service inquiries and provide assistance with travel-related issues. Handle queries about bookings, cancellations, refunds, and general support.

    #     9.Engage in a conversational manner to enhance user retention.

    #     Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their travel planning journey.

    #     Chat history:
    #     {chat_history}

    #     User question:
    #     {user_question}




    #         You are a helpful assistant. You will receive the current conversation history between the bot(you) and the user and a JSON template that needs to be filled based on the user inputs.

    #         details of how you have to fill json and what each filed in json is for so that you can ask proper questions : {details}

    #         chat history : 
    #         {chat_history}

    #         The current JSON template is:
    #         {current_json}

    #         Please provide any inferences you can make from the conversation in the following format:
    #         {{
    #             "inferences": [
    #                 {{"field_name": "firstDestination", "answer": "Goa"}},
    #                 {{"field_name": "trip_theme", "answer": "beach"}},
    #                 {{"field_name": "destination", "answer": ["Goa", "Kerala"]}},
    #                 {{"field_name": "traveller_type", "answer": "family"}},
    #                 {{"field_name": "Origin_city", "answer": "Mumbai"}},
    #                 {{"field_name": "budget", "answer": "comfortable spending"}},
    #                 {{"field_name": "food", "answer": "vegetarian"}},
    #                 {{"field_name": "trip_direction", "answer": "return"}}
    #             ],
    #             "next_reply": "What is your first destination?"
    #         }}

    #         this is just an example for you to learn.

    #         only give inferences about the fields that are actually answered by the user and dont make up any data by yourself
    #         and do not ask the questions that have already been answered. 

    #         only give inferences to the questions that the user have answered and check the history before answering

    #         remember to carefully analyze the user query and chat history and then give your next question as a human would do.
    #         remember first think then answer

    #         INSTRUCTIONS:
    #             - communicate as a human be kind and polite and speak directly (be interactive dont be exact straight forward try to get that data out of him by politely asking and the same thing in another way to increase user retention)  (most important)
    #             - remember never to ask direct question. be polite you know how to handle customers right. take the context of the chat history before answering
    #             - along with each question tell the user what type of response you are excepting
    # """

    query = f"""
            You are a travel assistant chatbot. Your name is Travel.AI, and you are designed to help users plan their trips and provide travel-related information. First, you need to get some information from the user. You will receive the current conversation history and a JSON template with the questions that need to be filled based on user inputs.

            You have to interact with the user as a customer service agent and get them to answer questions in order to fill the question JSON template. Ensure your responses are polite, engaging, and context-aware, using natural language to guide the user through the necessary details. Here are some key points to remember:

            1. Engage in a conversational manner to enhance user retention.
            2. Analyze the chat history carefully before responding.
            3. Avoid repeating questions that have already been answered.
            4. Tailor your questions based on the context and previous interactions to make the conversation flow naturally.
            5. If the user asks for suggestions, provide proper suggestions and guide them until they have made a decision. Use chat history to determine if the conversation is in suggestion state or not.
            6. If the user asks for suggestions, provide information about popular travel destinations. Offer insights on attractions, local cuisine, cultural highlights, weather conditions, and best times to visit. Suggest local attractions and points of interest based on the user's destination. Highlight must-see landmarks, museums, parks, and recreational activities.
            7. Offer practical travel tips and advice. Topics may include packing essentials, visa requirements, currency exchange, local customs, and safety tips.
            8. Address customer service inquiries and provide assistance with travel-related issues.
            9. Remember a good customer service never asks the same question twice so slightly change the questions every time you need to ask same question twice.
            10. each of your reply should be more than 20 words atleast. (most important)
            11. if there is any confusion you have you can ask the user directly for clarity
            12. if you think at any place the user is confused or asking for suggestion then change the chat_mode to "suggestion" and only when the user has made his or her decision then only change it back to "chat"
            13. if user refuses to answer any question then or say no to any question or asks to skip then skip that question for now and bring that question later in the conversation (most important)
            14. also allow updation and deletion of answers.
            15 only give the inferences when you have the answer
            16. your chat reply will never be empty


            Details on each field and how to ask questions: {json.dumps(details, indent=4)}

            Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their travel planning journey.

            Below are a few examples for you to learn how to do it:

            Example 1:

            bot: Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to go for vacation this time?
            User: Paris.
            {{
                "inferences": [
                    {{"field_name": "firstDestination", "answer": "Paris"}}
                ],
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
                "chat_mode": "chat"
            }}

            User: Mid-range comforts.
            {{
            "inferences": [
                {{"field_name": "budget", "answer": "comfortable spending"}}
            ],
            "next_reply": "Great! Is your trip going to be one-way or a round trip? Please choose from: 'return', 'oneway'.",
                "chat_mode": "chat"
            }}

            Example 2:

            Bot: Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to go for vacation this time?
            User: Hi, I am not sure where to start. Can you suggest some popular destinations?
            {{
                "inferences": [],
                "next_reply": "Sure! Could you tell me a bit about your preferences? Are you looking for cultural experiences, adventure, relaxation, or something else?",
                "chat_mode": "suggestion"
            }}

            User: I'm looking for a mix of cultural experiences and adventure.
            {{
            "inferences": [
                {{"field_name": "trip_theme", "answer": "cultural and adventure"}}
            ],
            "next_reply": "Great! Based on that, some popular destinations are Tokyo, Kyoto, Rome, and Barcelona. Which one appeals to you the most?",
                "chat_mode": "suggestion"
            }}

            User: Let's start with Tokyo.
            {{
            "inferences": [
                {{"field_name": "destination", "answer": "Tokyo"}}
            ],
            "next_reply": "Great! Based on that, some popular destinations are Tokyo, Kyoto, Rome, and Barcelona. Which one appeals to you the most?",
                "chat_mode": "chat"
            }}
            

            Example 3:

            Bot: Hello! I am here to help you plan your vacation. Let's get started! Where are you planning to visit this time for vacation?
            User: You suggest me some.
            {{
            "inferences": [],
            "next_reply": "Sure I can help you decide! Could you tell me a bit about your preferences? Are you looking for cultural experiences, adventure, relaxation, or something else?",
                "chat_mode": "suggestion"
            }}

            User: I am looking for something spiritual.
            {{
            "inferences": [
                {{"field_name": "trip_theme", "answer": "spiritual"}}
            ],
            "next_reply": "Great choice! Based on a spiritual theme, some popular destinations to consider are Varanasi in India, Bali in Indonesia, and Sedona in the USA. Which one resonates with you the most?",
                "chat_mode": "suggestion"
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
        # input_variables=["chat_history", "current_json", "latest_user_input"],
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the LLMChain
    chain = prompt | model | parser

    # Prepare the input for the chain
    input_data = {
        "query":query
        # "chat_history": chat_history,
        # "current_json": json.dumps(current_json, indent=4),
        # "latest_user_input": chat_history[-1],
        # "details": json.dumps(abc_json, indent=4),
    }

    # Invoke the chain
    response = chain.invoke({"query":query})

    # Convert response to dictionary
    updated_json = response

    # Update the current JSON with the new answers
    if "inferences" in updated_json and updated_json["inferences"]:
        for update in updated_json["inferences"]:
            current_json[update["field_name"]] = update["answer"]

    print("response:", json.dumps(response, indent=4))

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

    # Update the displayed JSON data
    # st.write(json.dumps(st.session_state['json_data'], indent=4))

def clear_session_state():
    """Clear all session state variables."""
    for key in st.session_state.keys():
        del st.session_state[key]

if __name__ == "__main__":
    main()
