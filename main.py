import json
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# credit: https://www.youtube.com/watch?v=9y9YYhCuLro&list=PLxFYCJW9AKJtoGL2-OwHQU7G3q1-I9KlP


def openai_init():
    client = OpenAI(api_key=open("api.key", "r").read())
    return client


def get_stock_price(ticker: str) -> str:
    return str(yf.Ticker(ticker).history(period="1y").iloc[-1]["Close"])


def calculate_SMA(ticker: str, window: str) -> str:
    data = yf.Ticker(ticker).history(period="1y")["Close"]
    return str(data.rolling(window=int(window)).mean().iloc[-1])


def calculate_EMA(ticker: str, window: str) -> str:
    data = yf.Ticker(ticker).history(period="1y")["Close"]
    return str(data.ewm(span=int(window), adjust=False).mean().iloc[-1])


def plot_stock_price(ticker: str):
    data = yf.Ticker(ticker).history(period="1y")
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"])
    plt.title(f"{ticker} Stock Price over last year")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig("stock_price.png")
    plt.close()


functions = [
    {
        "name": "get_stock_price",
        "description": "Returns the current stock price of the given ticker symbol for a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for the company (for example, AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_SMA",
        "description": "Returns the simple moving avarage for a given stock ticker and a window",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for the company (for example, AAPL for Apple).",
                },
                "window": {
                    "type": "integer",
                    "description": "The time frame to consider whan calculating SMA",
                },
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "calculate_EMA",
        "description": "Returns the exponential moving avarage for a given stock ticker and a window",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for the company (for example, AAPL for Apple).",
                },
                "window": {
                    "type": "integer",
                    "description": "The time frame to consider whan calculating EMA",
                },
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "plot_stock_price",
        "description": "Plots the stock price of the given ticker symbol for a company over the last year.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for the company (for example, AAPL for Apple).",
                }
            },
            "required": ["ticker"],
        },
    },
]

available_functions = {
    "get_stock_price": get_stock_price,
    "calculate_SMA": calculate_SMA,
    "calculate_EMA": calculate_EMA,
    "plot_stock_price": plot_stock_price,
}


MESSAGE_STORE = "messages"


def main():
    client = openai_init()

    # This is where we will store the messages
    if MESSAGE_STORE not in st.session_state:
        st.session_state[MESSAGE_STORE] = []

    st.title("Stock Analysis Chatbot Assistant")

    user_input = st.text_input("Enter your message here: ")

    # main processing loop
    if user_input:
        try:
            st.session_state[MESSAGE_STORE].append(
                {"role": "user", "content": f"{user_input}"}
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state[MESSAGE_STORE],
                functions=functions,
                function_call="auto",
            )

            response_message = response.choices[0].message

            if response_message.function_call:
                function_name = response_message.function_call.name
                function_args = json.loads(response_message.function_call.arguments)

                if function_name in ["get_stock_price", "plot_stock_price"]:
                    args_dict = {"ticker": function_args["ticker"]}
                elif function_name in ["calculate_SMA", "calculate_EMA"]:
                    args_dict = {
                        "ticker": function_args["ticker"],
                        "window": function_args["window"],
                    }

                function_to_call = available_functions[function_name]
                response = function_to_call(**args_dict)

                if function_name == "plot_stock_price":
                    st.image("stock_price.png")
                else:
                    st.session_state[MESSAGE_STORE].append(response_message)
                    st.session_state[MESSAGE_STORE].append(
                        {"role": "function", "name": function_name, "content": response}
                    )

                second_response_message = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state[MESSAGE_STORE],
                )

                st.markdown(second_response_message.choices[0].message.content)

                st.session_state[MESSAGE_STORE].append(
                    {
                        "role": "assistant",
                        "content": second_response_message.choices[0].message.content,
                    }
                )
            else:
                st.text(response_message.content)

                st.session_state[MESSAGE_STORE].append(
                    {"role": "assistant", "content": response_message.content}
                )
        except Exception as e:
            st.text(e)


if __name__ == "__main__":
    main()
