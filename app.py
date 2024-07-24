import os
import openai
import json
import gradio as gr
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()




openai.api_key = os.getenv("OPENAI_API_KEY")

neo4j_url = "neo4j+s://" + str(os.getenv("NEO4J_URL"))
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
driver = GraphDatabase.driver(neo4j_url, auth=AUTH)







## MAIN FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

#standard API Call to open AI with system prompt and user prompts.
def chat(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0):
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
            model = "gpt-4o-mini", 
            response_format = {"type": "json_object"},
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": storyline_prompt}],
            temperature=0
            )
     res = response.choices[0].message.content
     map = json.loads(res)
     #pretty_list = "\n".join([f"⚡ {key}: {value}" for key, value in map.items()])
     storypoint_name_list = [map[key] for key in map]
     storypoint_name_nested = [storypoint_name_list]
     storypoint_name_nested = list(zip(*storypoint_name_nested))
     #add one column to the list with the name of the storypoint

     storypoint_name_nested = [[f"SP {i}", item] for i, item in enumerate(storypoint_name_nested, 1)]

     return map, storypoint_name_nested

#we need this function to turn the non iterable nested list that is gr.List into a simple list.
def iterator_for_gr(nested_list, i):
     #Initialize a variable to store the processing result
     storypoint_names = []

     #since the gr.List is a List[List] (nested list, we need to unwrap the 0th element)
     for item in nested_list[0]:
         storypoint_names.append(item)
         

     # Return a string that combines all the processed results
     return str(storypoint_names[i-1])



# ---------------------------------------------------------------------------------------------------------------------
## Calculate Input Storypoints Similarity to Storypoints in Database

from openai import OpenAI
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))




def get_embedding_inputstorypoints(storyline_output_storypoint_name_list, model="text-embedding-3-large"):

    #input has 2 colums, pick the second column
    storyline_output_storypoint_name_list = [[item[1]] if type(item[1]) is not list else item[1] for item in storyline_output_storypoint_name_list if len(item) > 1]

    # transform storyline_output_storypoint_name_list to pandas dataframe
    input_storypoints = pd.DataFrame(storyline_output_storypoint_name_list, columns=['description'])
    # get embeddings for input storypoints
    input_storypoints['ada_embedding'] = input_storypoints.description.apply(lambda x: client.embeddings.create(input = [x], model=model).data[0].embedding)
    
    return input_storypoints


# Function to fetch embeddings from Neo4j
def fetch_embeddings(driver):
    query = """
    MATCH (sp:STORYPOINT)
    RETURN sp.id AS id, sp.embedding AS embedding
    """
    embeddings = {}
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            embeddings[record['id']] = np.array(record['embedding'])
    return embeddings

# Function to calculate cosine similarity and find the highest similarities
def find_highest_similarities(existing_embeddings, new_embeddings):
    # Transform embeddings into arrays for the calculation
    existing_ids, existing_vecs = zip(*existing_embeddings.items())
    new_ids, new_vecs = zip(*new_embeddings.items())
    existing_vecs = np.array(existing_vecs)
    new_vecs = np.array(new_vecs)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(new_vecs, existing_vecs)

    # Find the index with the highest similarity for each new embeddin TODO: Replace with top 5 most similar
    max_indices = np.argmax(similarity_matrix, axis=1)
    similarities = np.max(similarity_matrix, axis=1)


    # Pair each new storypoint with the existing one that has the highest similarity
    highest_pairs = [(new_ids[i], existing_ids[max_indices[i]], similarities[i]) for i in range(len(new_ids))]
    return highest_pairs

def coordinate_simcalculation(storyline_output_storypoint_name_list):

    # Fetch existing embeddings from Neo4j
    existing_embeddings = fetch_embeddings(driver)
    input_storypoints = get_embedding_inputstorypoints(storyline_output_storypoint_name_list)
    # Assume new_embeddings come from your Python processing earlier
    new_embeddings = {row['description']: row['ada_embedding'] for index, row in input_storypoints.iterrows()}

    # Find highest similarities
    highest_similarities = find_highest_similarities(existing_embeddings, new_embeddings)

    # Display results
    for new_id, existing_id, similarity in highest_similarities:
        print(f"Input STORYPOINT '{new_id}' is most similar to existing STORYPOINT '{existing_id}' with a similarity of {similarity:.2f}")

    HTMLoutput = construct_hmtl(highest_similarities)

    return HTMLoutput, highest_similarities

def construct_hmtl(highest_similarities, nodes_to_show=["SLIDE_DECK", "SLIDE", "STORYPOINT"]):

    storypoint_ids = [existing_id for _, existing_id, _ in highest_similarities]
    print(storypoint_ids)


    # Starting with the base of the query
    query_parts = [
        f"WITH {storypoint_ids} AS ids",
        "MATCH (sp:STORYPOINT) WHERE sp.id IN ids",
        "WITH sp",
        "ORDER BY apoc.coll.indexOf(ids, sp.id)",
        "WITH COLLECT(sp) AS sps",
        "UNWIND RANGE(0, SIZE(sps) - 2) AS idx",
        "WITH sps, sps[idx] AS sp_start, sps[idx + 1] AS sp_end",
        "CALL apoc.create.vRelationship(sp_start, 'FOLLOWS', {}, sp_end) YIELD rel",
        "WITH sps, sp_start, rel, sp_end",
        "UNWIND sps AS sp"
    ]

    # Initialize the match and return parts of the query
    match_parts = []
    return_parts = []

    # Include virtual relationship and its nodes conditionally
    if "STORYPOINT" in nodes_to_show:
        return_parts.extend(["sp_start", "rel", "sp_end", "sp"])
    
    # Conditionally add SLIDE and SLIDE_DECK with their relationships
    if "SLIDE" in nodes_to_show or "SLIDE_DECK" in nodes_to_show:
        match_parts.append("(sp)<-[r1:ASSIGNED_TO]-(s:SLIDE)")
        return_parts.extend(["s", "r1"])
        if "SLIDE_DECK" in nodes_to_show:
            match_parts.append("<-[r2:CONTAINS]-(sd:SLIDE_DECK)")
            return_parts.extend(["sd", "r2"])

    # Construct the final query
    query = "\n".join(query_parts)
    if match_parts:
        query += "\nMATCH " + "".join(match_parts)
    if return_parts:
        query += "\nRETURN " + ", ".join(return_parts)
    else:
        query += "\nRETURN 'No nodes to show based on the selected types'"
    
    

    graphVisualHTML = f"""

<head>
    <title>DataViz</title>
    <style type="text/css">
        body {{
            display: flex; /* Use flexbox to align children */
            justify-content: center; /* Center horizontally in the flex container */
            align-items: center; /* Center vertically if desired */
            min-height: 100vh; /* Ensure the body takes at least the height of the viewport */
            margin: 0; /* Remove default margin */
        }}
        #viz {{
            width: 1200px;
            height: 700px;
            background-color: #f0f0f0; /* Lighter grey background for the viz div */
            padding: 10px; /* Adds padding inside the div */
        }}
        .heading {{
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }}
        #queryCypher {{
            opacity: 0;
        }}
    </style>
</head>
<body>

    

    <div id="viz">
        <p id="queryCypher">{query}</p>
    </div>
    
    <div class="custom-menu" style="display: none; position: absolute; z-index: 1000; background: white; border: 1px solid #ccc; padding: 5px; box-shadow: 2px 2px 5px #888;">
        <ul>
            <li onclick="alert('Action 1')">Action 1</li>
            <li onclick="alert('Action 2')">Action 2</li>
        </ul>
    </div>
</body>
    """
    return graphVisualHTML

scripts = """

async () => {
 
    const script = document.createElement("script");
    script.src = "https://rawgit.com/neo4j-contrib/neovis.js/master/dist/neovis.js";
    document.head.appendChild(script);

    globalThis.draw = (queryCypher) =>{
     
        var config = {
                containerId: "viz",
                neo4j: {
                    serverUrl: "bolt://"""+os.getenv("NEO4J_URL")+""":7687",
                    serverUser: \""""+os.getenv("NEO4J_USERNAME")+"""\",
                    serverPassword: \""""+os.getenv("NEO4J_PASSWORD")+"""\",
                    driverConfig: {
                        encrypted: "ENCRYPTION_ON",
                        trust: "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES",
                    },
                },
                labels: {
                    SLIDE: {
                        [NeoVis.NEOVIS_ADVANCED_CONFIG]: {
                        static: {
                            shape: "image" // Sets the shape to use an image (use "circularImage" for circular nodes)
                        },
                        function: {
                            image: (node) => "https://slidestorage.s3.eu-north-1.amazonaws.com/" + node.properties.object_id + ".png"
                        }
                        }
                    },
            STORYPOINT: {
                label: "description",
                [NeoVis.NEOVIS_ADVANCED_CONFIG]: {
                    static: {
                        caption: "description",
                        shape: 'box',
                        color: {
                            background: 'white',
                            border: 'lightgray',
                            highlight: {
                                background: 'lightblue',
                                border: 'blue'
                            }
                        },
                        font: {
                            color: 'black',
                            size: 14, // Pixel size
                            face: 'Helvetica' // Uniform font across all graph elements
                        }
                    }
                }
            },
            SLIDE_DECK: {
                label: "title",
                [NeoVis.NEOVIS_ADVANCED_CONFIG]: {
                    static: {
                        caption: "title",
                        shape: 'circle', // Updated to circle for a uniform and standard appearance
                        color: {
                            background: 'lightyellow',
                            border: 'gold',
                            highlight: {
                                background: 'yellow',
                                border: 'darkorange'
                            }
                        },
                        font: {
                            color: 'black',
                            size: 14, // Pixel size
                            face: 'Helvetica' // Uniform font across all graph elements
                        }
                    }
                }
            }

            },

			relationships: {
				CONTAINS: {
					[NeoVis.NEOVIS_ADVANCED_CONFIG]: {
						static: {
							label: "Contains",
							thickness: 2, // Enhanced thickness for better visibility
							color: '#34495e', // Deep, neutral blue color for a modern look
							font: {
								color: '#2c3e50', // Dark grey color for strong contrast against light background
								size: 14, // Larger font size for enhanced readability
								face: 'Helvetica' // Modern font for a clean appearance
							},
							dashes: false, // Solid line to indicate a strong, permanent relationship
						}
					}
				},
				ASSIGNED_TO: {
					[NeoVis.NEOVIS_ADVANCED_CONFIG]: {
						static: {
							label: "Assigned To",
							thickness: 2, // Consistent thickness across all relationship types
							color: '#16a085', // Distinctive teal color to differentiate from 'CONTAINS'
							font: {
								color: '#2c3e50', // Dark grey to maintain visibility and consistency
								size: 14,
								face: 'Helvetica'
							},
							arrows: {
								to: { enabled: true, scaleFactor: 1.2 } // Prominent arrow for visual emphasis
							},
						}
					}
				},
				FOLLOWS: {
					[NeoVis.NEOVIS_ADVANCED_CONFIG]: {
						static: {
							label: "Follows",
							thickness: 2,
							color: '#8e44ad', // Soft purple for visual distinction
							font: {
								color: '#2c3e50', // Dark grey to ensure readability on light backgrounds
								size: 14,
								face: 'Helvetica'
							},
							arrows: {
								to: { enabled: true, scaleFactor: 1.5 } // Larger arrow to denote directionality
							},
							dashes: true // Dashed line to indicate a temporal or less permanent relationship
						}
					}
				},
			},

            
            visConfig: {
                layout: {
                    improvedLayout: true,
                    //hierarchical: true,
                    clusterThreshold: 7,
                },
            },
            initialCypher: queryCypher,
        };

        console.log("Drawing visualization");
        var viz = document.getElementById("viz");

        if (!viz) {
            console.error("Visualization container not found.");
            return;
        }      

        try {
            viz = new NeoVis.default(config);
            viz.render();
            viz.registerOnEvent("completed", () => {
                viz.network.on("oncontext", function (params) {
                    params.event.preventDefault();
                    const customMenu = document.querySelector('.custom-menu');
                    
                    if (customMenu) {
                    console.log("Displaying custom menu.");
                        const containerRect = document.getElementById('viz').getBoundingClientRect();
                        customMenu.style.display = 'block';
                        customMenu.style.top = `${params.event.pageY - containerRect.top + window.scrollY}px`;
                        customMenu.style.left = `${params.event.pageX - containerRect.left + window.scrollX}px`;
                    }
                });
            });

            
        } catch (error) {
            console.error('Error rendering NeoVis:', error);
        }
        

        
    }

    script.onload = () => {
        console.log("NeoVis.js loaded"); 
        //draw();
    };


}
"""



js_click = """
<script>

// Function to handle the mutations
function handleMutations(mutations) {
    for (let mutation of mutations) {
        if (mutation.type === 'childList') {
            const drawElement = document.getElementById('viz');
            if (drawElement) {
                console.log('Call draw function.');
                var element = document.getElementById('queryCypher');  // Access the element by its ID
                var text = element.innerText; 
                draw(text);
                // Disconnect the observer after clicking the drawElement is found
                //observer.disconnect();
                return;
            }
        }
    }
}

// Create a new MutationObserver instance
const observer = new MutationObserver(handleMutations);

// Configuration of the observer:
const config = {
    childList: true,  // Observe direct children
    subtree: true,    // Observe all descendants
    attributes: false // Do not observe attribute changes
};

// Start observing the body for configured mutations
observer.observe(document.body, config);

console.log("Observer is set to monitor changes in the document body.");


</script>
"""

css = """
#SPList {
    font-family: 'Arial', sans-serif;
    background-color: #f8f9fa;
    color: #333;
    background-color: #ffffff;
    color: #333;
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 10px;
    margin: 5px;
}
#SPList .gr-array-container {
    gap: 10px;
}"""


## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
graphVisual = gr.HTML()
highest_similarities_gradio_list = gr.List(type="array", interactive=False, visible=False)
nodeSelector = gr.Dropdown(label="Filter nodes", choices=["SLIDE_DECK", "SLIDE", "STORYPOINT"], value=["SLIDE_DECK", "SLIDE", "STORYPOINT"], multiselect=True, scale=1)
filterBTN = gr.Button("Apply Filter")

with gr.Blocks(title='Slide Inspo', js=scripts, head = js_click, theme = gr.themes.Monochrome(), css= css).queue(default_concurrency_limit=1) as demo:
    
    highest_similarities_gradio_list.render()
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 1. Input: 🔍")
            storyline_prompt = gr.Textbox(placeholder = 'Give us a topic and we will provide a storyline for you! For example: "SCRUM in Software Development"', 
                                        label = 'Topic to build:',
                                        lines=5,
                                        scale = 3)
            nr_storypoints_to_build = gr.Number(value=5,
                                        label="How many storypoints?",
                                        scale =1)
            storyline_output_JSON = gr.JSON(visible=False)
            
            btn = gr.Button("Build Storyline 🦄")

        with gr.Column(scale=1):
            gr.Markdown("# 2. Storyline: 🦄")
            storyline_output_storypoint_name_list = gr.List(visible=True, type="array", interactive=True, label="Adapt and add Storypoints, if needed: 📝", scale=1, wrap=True, col_count=[2, "fixed"], elem_id="SPList", headers=["#SP", "Description"])                
            #storyline_output_pretty = gr.Textbox(label="Your Storyline:", lines=13, scale=3, interactive=False)
            submit_button = gr.Button("⚡ Find Slides ⚡", elem_id="visGraph")
            submit_button.click(fn= coordinate_simcalculation, inputs=[storyline_output_storypoint_name_list], outputs=[graphVisual, highest_similarities_gradio_list]).then(js = js_click)



            btn.click(slide_deck_storyline, 
                                    inputs = [storyline_prompt, nr_storypoints_to_build], 
                                    outputs = [storyline_output_JSON, storyline_output_storypoint_name_list])
                
            storyline_prompt.submit(slide_deck_storyline, 
                                    inputs = [storyline_prompt, nr_storypoints_to_build], 
                                    outputs = [storyline_output_JSON, storyline_output_storypoint_name_list])


    with gr.Row():
        with gr.Group():
            with gr.Column(scale=1):
                nodeSelector.render()
            with gr.Column(scale=1):
                filterBTN.render()
                filterBTN.click(fn= construct_hmtl, inputs=[highest_similarities_gradio_list, nodeSelector], outputs=[graphVisual]).then(js = js_click)

            with gr.Row():
                graphVisual.render()



    

gr.close_all()
demo.launch()