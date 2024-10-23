import os
from dotenv import load_dotenv

# Add Azure OpenAI package
from openai import AzureOpenAI

def summary(caption): 
    try: 
        # Get configuration settings 
        load_dotenv()
        azure_oai_endpoint ="https://openai0878675.openai.azure.com/"
        azure_oai_key = "2558fb7b74aa49f58474e5a160a99fd3"
        azure_oai_deployment = "gpt-35-turbo-16k"
        
        # Initialize the Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=azure_oai_endpoint, 
            api_key=azure_oai_key,  
            api_version="2024-02-15-preview"
        )        

        # Create a system message (you can modify this to encourage longer responses)
        system_message = "You are a story generator. Write a detailed story based on the caption provided."
        
        # Initialize messages array
        messages_array = [{"role": "system", "content": system_message}]
        
        # Provide the caption and ask for a longer story (around 400-500 words)
        input_text = f"Generate a detailed story in around 400-500 words based on this caption: {caption}"
        
        print("\nSending request for generating story to Azure OpenAI ...\n\n")
        
        # Add user input to the messages array
        messages_array.append({"role": "user", "content": input_text})
        
        # Send request to Azure OpenAI model with increased max_tokens
        response = client.chat.completions.create(
            model=azure_oai_deployment,
            temperature=0.7,
            max_tokens=700,  # Adjust this for ~400-500 word story (600-700 tokens)
            messages=messages_array
        )
        
        # Extract the generated story from the response
        generated_text = response.choices[0].message.content
        
        # Add the generated story to the messages array
        messages_array.append({"role": "assistant", "content": generated_text})
        
        # Return the generated story
        print(generated_text)
        return generated_text

    except Exception as ex:
        # Return the error message for display in Streamlit
        return f"Error generating story: {str(ex)}"
