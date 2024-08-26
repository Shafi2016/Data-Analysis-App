import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Inject custom CSS to change font size and background color
st.markdown(
    """
    <style>
    .big-font {
        font-size: 20px !important;
    }
    .dataframe {
        font-size: 18px !important;
    }
    .custom-container {
        background-color: #0077a8;
        padding: 10px;
        border-radius: 5px;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
    }
    .custom-container h1 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Initialize the Streamlit app
st.markdown('<div class="custom-container"><h1>Data Analysis App</h1></div>', unsafe_allow_html=True)

# Move the file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Check if the file is CSV or Excel
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("## Data Preview")
    st.markdown(df.head().to_html(classes="dataframe"), unsafe_allow_html=True)  # Apply custom CSS to DataFrame

    # Initialize the LangChain agent with the provided API key
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key),
        df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    # Create a text input for the query
    query = st.text_input("Enter your query (e.g., 'How many rows are there?')")

    if st.button("Run Query"):
        # Invoke the agent with the query and get the response
        response = agent.run(query)

        # Check if a plot was generated
        current_fig = plt.gcf()  # Get the current figure (if any)

        if current_fig.get_axes():  # Check if the figure has any axes (indicating a plot)
            st.pyplot(current_fig)  # Display the plot
            plt.clf()  # Clear the figure to avoid conflicts with future plots
        else:
            # Display only the answer (assuming response is a dictionary or JSON-like structure)
            if isinstance(response, dict) and "output" in response:
                st.markdown(
                    f'<div class="custom-container"><p class="big-font">{response["output"]}</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                # Fallback to displaying the entire response if it's not in the expected format
                st.markdown(
                    f'<div class="custom-container"><p class="big-font">{response}</p></div>',
                    unsafe_allow_html=True,
                )

else:
    st.write("Please upload a file to begin.")
