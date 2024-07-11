import os
import openai
import json
import gradio as gr
from neo4j import GraphDatabase





openai.api_key = os.getenv("OPENAI_API_KEY")

neo4j_url = os.getenv("NEO4J_URL")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
#driver = GraphDatabase.driver(neo4j_url, auth=AUTH)





## MAIN FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

#standard API Call to open AI with system prompt and user prompts.
def chat(system_prompt, user_prompt, model="gpt-3.5-turbo-1106", temperature=0):
    response = openai.chat.completions.create(
        model = model,

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
            temperature=temperature
            )
    res = response.choices[0].message.content
      
    return res

# NOT USED IN THIS DEMO
# this function formats the user input and links it to the chat history for further context awareness.
def format_chat_prompt(message, chat_history, max_convo_length):
    prompt = ""
    for turn in chat_history[-max_convo_length:]:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt




#this is a simple prompt that takes a storyline prompt and formats an output in json to return a storyline of X slides.
def slide_deck_storyline(storyline_prompt, nr_of_storypoints=5):
     nr_of_storypoints = str(nr_of_storypoints)
     system_prompt = f"""You are an AI particularly skilled at captivating storytelling for educational purposes.
                        You know how tell a compelling, structure and exhaustive narrative around any given academic topic.
                        What you are particularly good at, is taking any given input and building a storyline in the delivered as
                        {nr_of_storypoints} storypoints and nothing else. This is your only chance to impress me.
                        
                        You will recieve a topic and you will answer with a list of {nr_of_storypoints} crucial storypoints.
                        
                        Instrucitions:
                        Give me a json map of {nr_of_storypoints} storypoints that you would include in a slide deck about {storyline_prompt}.
                        Only answer with the list. Do not include any nicities, greetings or repeat the task.
                        Never make more than {nr_of_storypoints} storypoints. This is important!
                        Just give me the list. Keep the list concise and only answer with the list in this format.
                        Name every key a storypoint (Storypoint 1, Storypoint 2 ... Storypoint N).
                        The elements of the list should be storypoints, highlighting what the point the sliStorypoint is trying to make is.
                        """

     response = openai.chat.completions.create(
            model = "gpt-3.5-turbo-1106", 
            response_format = {"type": "json_object"},
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": storyline_prompt}],
            temperature=0
            )
     res = response.choices[0].message.content
     map = json.loads(res)
     pretty_list = "\n".join([f"⚡ {key}: {value}" for key, value in map.items()])
     storypoint_name_list = [map[key] for key in map]
     storypoint_name_nested = [storypoint_name_list]
     return map, storypoint_name_nested, pretty_list

#we need this function to turn the non iterable nested list that is gr.List into a simple list.
def iterator_for_gr(nested_list, i):
     #Initialize a variable to store the processing result
     storypoint_names = []

     #since the gr.List is a List[List] (nested list, we need to unwrap the 0th element)
     for item in nested_list[0]:
         storypoint_names.append(item)
         

     # Return a string that combines all the processed results
     return str(storypoint_names[i-1])



     

## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------

with gr.Blocks(title='Slide Inspo', theme='Soft') as demo:
     with gr.Row():
          with gr.Column(scale=1):
                gr.Markdown("# 1. Input: 🔍")
                storyline_prompt = gr.Textbox(placeholder = 'Give us a topic and we will provide a storyline for you! For example: "Risk Management in Venture Capital"', 
                                            label = 'Topic to build:',
                                            lines=5,
                                            scale = 3)
                nr_storypoints_to_build = gr.Number(value=5,
                                                label="How many storypoints?",
                                                scale =1)
                storyline_output_JSON = gr.JSON(visible=False)
                storyline_output_storypoint_name_list = gr.List(visible=True, type="array")
                btn = gr.Button("Build Storyline 🦄")

          with gr.Column(scale=1):
               gr.Markdown("# 2. Storyline: 🦄")
                            
               storyline_output_pretty = gr.Textbox(label="Your Storyline:", lines=13, scale=3)
               submit_button = gr.Button("⚡ Find Slides ⚡")

               btn.click(slide_deck_storyline, 
                                        inputs = [storyline_prompt, nr_storypoints_to_build], 
                                        outputs = [storyline_output_JSON, storyline_output_storypoint_name_list, storyline_output_pretty])
                
               storyline_prompt.submit(slide_deck_storyline, 
                                        inputs = [storyline_prompt, nr_storypoints_to_build], 
                                        outputs = [storyline_output_JSON, storyline_output_storypoint_name_list, storyline_output_pretty])



gr.close_all()
demo.launch(share=True)