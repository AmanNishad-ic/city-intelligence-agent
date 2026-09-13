from dotenv import load_dotenv
import os
import requests
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
load_dotenv()

#now let create some tools

#weather tool
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
@tool
def get_weather(city : str) -> str:
    '''Get current weather of a city'''
    if not OPENWEATHER_API_KEY:
        return {
            "success": False,
            "error": "OPENWEATHER_API_KEY is missing from the .env file."
        }

    params = {
        "q": f"{city},IN",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(
            WEATHER_URL,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()

        return {
            "success": True,
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature_celsius": data["main"]["temp"],
            "feels_like_celsius": data["main"]["feels_like"],
            "condition": data["weather"][0]["description"],
            "humidity_percent": data["main"]["humidity"],
            "wind_speed_mps": data["wind"]["speed"]
        }

    except requests.exceptions.HTTPError:
        if response.status_code == 401:
            message = "Invalid or inactive OpenWeather API key."
        elif response.status_code == 404:
            message = f"City '{city}' was not found."
        elif response.status_code == 429:
            message = "OpenWeather API request limit exceeded."
        else:
            message = f"OpenWeather returned HTTP {response.status_code}."

        return {
            "success": False,
            "error": message
        }

    except requests.exceptions.RequestException as error:
        return {
            "success": False,
            "error": f"Unable to contact OpenWeather: {error}"
        }

#Tavily news tool
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError(
        "TAVILY_API_KEY is missing. Please add it to your .env file."
    )

tavily_client = TavilyClient(api_key=tavily_api_key)


@tool
def get_news(city: str) -> str:
    """
    Get the latest news about an Indian city.

    Use this tool when the user asks for recent news,
    events, politics, infrastructure, crime, business,
    sports, or other updates about a city in India.
    """

    city = city.strip()

    if not city:
        return "Please provide a valid city name."

    query = f"latest important news and local updates from {city}, India"

    try:
        response = tavily_client.search(
            query=query,
            topic="news",
            search_depth="basic",
            max_results=5,
            time_range="week",
            include_published_date=True,
            filter_by_published_date=True
        )

        results = response.get("results", [])

        if not results:
            return f"No recent news was found for {city}."

        formatted_news = []

        for index, article in enumerate(results, start=1):
            title = article.get("title", "No title")
            content = article.get("content", "No summary available")
            url = article.get("url", "")
            published_date = article.get(
                "published_date",
                "Publication date unavailable"
            )

            news_item = (
                f"{index}. {title}\n"
                f"Published: {published_date}\n"
                f"Summary: {content}\n"
                f"Source: {url}"
            )

            formatted_news.append(news_item)

        return f"Latest news for {city}:\n\n" + "\n\n".join(formatted_news)

    except Exception as error:
        return f"Unable to retrieve news for {city}: {error}"



#Model LLM

# llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro-0813")

# model = ChatHuggingFace(llm=llm)

model = ChatMistralAI(model="ministral-8b-2512")

tools = {
    "get_weather": get_weather,
    "get_news" : get_news
}

llm_with_tool = model.bind_tools([get_weather,get_news])

#Agent LOOP - Very important

messages = []

print("city intelligence System")

while True:
    user_input = input("You: ")
    if user_input.lower()== "exit":
        break
    messages.append(HumanMessage(content=user_input))
    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        # if tool is required
        
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]

            #HUMAN_IN THE LOOP
                confirm = input(f"Agent wants to call {tool_name} Approve (yes/no)")
                if confirm.lower() == "no":
                    print("tool call deniend and I cannot et the latest informetion")
                    break

            #esxcute tool

                tool_result = tools[tool_name].invoke(tool_call)

                messages.append(ToolMessage(
                    content= tool_result,
                    tool_call_id = tool_call['id']

                ))

            continue
        else:
            print(result.content)
            break     