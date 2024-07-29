import os
import openai
import json
import gradio as gr
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import re


load_dotenv()




openai.api_key = os.getenv("OPENAI_API_KEY")

neo4j_url = "neo4j+s://" + str(os.getenv("NEO4J_URL"))
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
driver = GraphDatabase.driver(neo4j_url, auth=AUTH)



user_id = None




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
            model = "gpt-4o", 
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


#this is a prompt that takes a filter prompt and formats an output in json to return a filter cypress query.
def custom_filtering(filter_prompt, current_cypher_query, neo4j_response):


     system_prompt = f"""You are an AI specifically trained to write accurate Neo4j Cypher queries.
                    This is your only chance to impress me.

                    In the Neo4j database, the nodes are defined as SLIDE_DECK, SLIDE, STORYPOINT, and AUTHOR connected by these relationships:
                    (sd:SLIDE_DECK)-[:CONTAINS]->(s:SLIDE)
                    (s:SLIDE)-[:ASSIGNED_TO]->(sp:STORYPOINT)
                    (sp1:STORYPOINT)-[:FOLLOWS]->(sp2:STORYPOINT)
                    (sd:SLIDE_DECK)-[:CREATED_BY]->(a:AUTHOR)

                    You will receive a the current cypher query and its corresponding Neo4j response. Your task is to respond with a new Cypher query that filters based on the user's request.
                    Do NOT forget to return relationships connecting the nodes if needed.

                    Instructions:
                    The current cypher query is: "{current_cypher_query}"
                    The Neo4j response is: "{neo4j_response}"

                    Ensure the correct STORYPOINT nodes in the order is adressed, as specified in the initial line of the current cypher query.
                    For example, in the sequence ['113', '-6555727423036779192A_outlier', '5554388242771153481A_outlier', '25', '1431557444396440005A_outlier'], '-6555727423036779192A_outlier' is the second STORYPOINT.

                    Respond with exactly a single JSON object containing the key "cypherquery" and the value of the requested query.
                    Do not include any nicities, greetings or repeat the task. Keep the query concise and only answer in this format.
                    """


     response = openai.chat.completions.create(
            model = "gpt-4o", 
            response_format = {"type": "json_object"},
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": filter_prompt}],
            temperature=0
            )
     res = response.choices[0].message.content
     res = json.loads(res)
     cypher_query = res["cypherquery"]

    # Enhanced pattern to catch variations including potential spaces, newlines, and mixed cases
     pattern = r"(?i)\b(CREATE|SET|DELETE|REMOVE|MERGE)\s*(\(|\[|\{)?"
    
    # Split the query into individual statements based on semicolons
     statements = cypher_query.split(';')
    
    # Further process each statement to check for conditional or nested writes
     def is_write_statement(statement):
        # Check if the statement includes write operations
        if re.search(pattern, statement):
            return True
        # Check for potentially hidden write operations within sub-queries or function calls
        nested_patterns = [
            r"FOREACH\s*\(([^)]+)\)",  # Looking inside FOREACH loops
            r"CASE\s+WHEN\s+[^:]+:\s+[^:]+ELSE\s+[^:]+END",  # Checking CASE statements
            r"CALL\s+[^()]+(\(.*\))?YIELD\s+[^()]+",  # Checking CALL statements
        ]
        for nested_pattern in nested_patterns:
            if re.search(nested_pattern, statement, re.IGNORECASE | re.DOTALL):
                # Recursively check inside the nested statement
                match = re.search(nested_pattern, statement, re.IGNORECASE | re.DOTALL)
                if match and is_write_statement(match.group(1)):
                    return True
        return False

    # Filter statements that contain write operations
     filtered_statements = [stmt for stmt in statements if not is_write_statement(stmt)]
    
    # Join the filtered statements back into a single query string
     filtered_query = '; '.join(filtered_statements)

     html = construct_hmtl(query = filtered_query)

     print(res["cypherquery"])
     print(filtered_query)
     
     return html, filtered_query






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

    HTMLoutput, query = construct_hmtl(highest_similarities)

    return HTMLoutput, highest_similarities, query

def track_user_interaction(user_input, action):
    global user_id
    
    user_id = str(user_id)
    connection = sqlite3.connect('/data/user_interactions.db')
    cursor = connection.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_input_str = str(user_input)
    action_str = str(action)
        # Escape single quotes by replacing them with two single quotes
    user_input_str = user_input_str.replace("'", "''")
    action_str = action_str.replace("'", "''")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_interactions (
        user_input TEXT NOT NULL,
        action TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        user_id TEXT NOT NULL
    );
    ''')    
    insert_query = "INSERT INTO user_interactions (user_input, action, timestamp, user_id) VALUES ('" + user_input_str + "', '" + action_str + "', '" + timestamp + "', '" + user_id + "')"

    cursor.execute(insert_query)
    connection.commit()
    connection.close()

def profile_user(username, password):
    global user_id
    user_id = username
    track_user_interaction("", "login")
    return True



def get_neo4j_response(query):
    with driver.session() as session:
        result = session.run(query)
        response = [record for record in result]
    return response

def construct_hmtl(highest_similarities = None, nodes_to_show=["SLIDE_DECK", "SLIDE", "STORYPOINT"], query=None):

    if query is None:
        
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
            /*width: 1600px;*/
            height: 700px;
            /*background-color: #f0f0f0;  Lighter grey background for the viz div */
            padding: 5px; /* Adds padding inside the div */
        }}
        .heading {{
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }}
        #queryCypher {{
            display:none;
        }}
    </style>
</head>
<body>

    

    <div id="viz">
        <p id="queryCypher">{query}</p>
    </div>
    

</body>
    """
    return graphVisualHTML, query

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
                            face: 'Quicksand' // Uniform font across all graph elements
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
                            face: 'Quicksand' // Uniform font across all graph elements
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
								face: 'Quicksand' // Modern font for a clean appearance
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
								face: 'Quicksand'
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
								face: 'Quicksand'
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
            //viz.registerOnEvent("completed", () => {
            //    viz.network.on("oncontext", function (params) {
            //        params.event.preventDefault();
            //        const customMenu = document.querySelector('.custom-menu');
            //        
            //        if (customMenu) {
            //        console.log("Displaying custom menu.");
            //            const containerRect = document.getElementById('viz').getBoundingClientRect();
            //            customMenu.style.display = 'block';
            //            customMenu.style.top = `${params.event.pageY - containerRect.top + window.scrollY}px`;
            //            customMenu.style.left = `${params.event.pageX - containerRect.left + window.scrollX}px`;
            //        }
            //    });
            //});

            
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



js_call_draw = """
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

# CSS for the Storypoint list
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
nodeSelector = gr.Dropdown(scale = 3, label="Filter nodes", choices=["SLIDE_DECK", "SLIDE", "STORYPOINT"], value=["SLIDE_DECK", "SLIDE", "STORYPOINT"], multiselect=True)
filterBTN = gr.Button("Apply Filter")

with gr.Blocks(title='Slide Inspo', js=scripts, head = js_call_draw, theme = gr.themes.Monochrome()).queue(default_concurrency_limit=1) as demo:

    highest_similarities_gradio_list.render()
    with gr.Row():
        gr.Markdown("# NarrativeNet Weaver")
    with gr.Row():    
        queryPlaceholder = gr.Textbox(visible=False)
        responsePlaceholder = gr.Textbox(visible=False)
        user_id = gr.Textbox(visible=False)
        customFilterQuery = gr.Textbox(visible=False)
        with gr.Column(scale=1):
            gr.Markdown("""## 1. Input: 🔍

                        **Define Your Workshop Objective.**
                        Choose a topic that is timely and fills a skill gap relevant to your consulting firm’s strategic goals. 
                        Define learning goals that focus on acquiring skills applicable in real-world consulting scenarios. 
                        Consider how mastering these skills can innovate and enhance your firm’s service offerings, aligning with emerging market needs and providing a competitive edge.  
                        *Our AI takes care to draft story points based on your input.*  
                        **What are Story Points?**
                        Story points are key milestones in your presentation that underline important learning outcomes. You can adapt them in the next step to cover skills and insights crucial for your firm’s services.
                        """)
            storyline_prompt = gr.Textbox(placeholder = """Give us a topic and we will provide a storyline for you! For example: 
                                          
Topic: AI enhancements for decision-making and process automation across multiple sectors such as finance, healthcare, and retail.

Goals: Equip participants with the ability to apply AI techniques practically to solve industry-specific challenges.

Implementation: Explore and integrate AI-driven solutions tailored to each sector, enhancing the firm’s service offerings.

Outcome: Fellow consultant will develop a comprehensive understanding of AI's potential and capabilities. This knowledge will enable consultants to effectively identify and propose AI-driven solutions.""", 
                                        label = 'Topic to build:',
                                        lines=5,
                                        scale = 3)
            nr_storypoints_to_build = gr.Number(value=3,
                                        label="How many story points?",
                                        scale =1)
            storyline_output_JSON = gr.JSON(visible=False)
            
            btn_buildstoryline = gr.Button("Build Storyline 🦄")

        with gr.Column(scale=1):
            gr.Markdown("""## 2. Storyline: 🦄

                **Content Requirements and Story Points.**
                Develop content that supports your workshop’s learning goals, using theories, case studies, and real-world applications.                             
                **Evaluating Story Points.**
                Effective story points are clear, engaging, and directly tied to your objectives. They should advance understanding and skill acquisition.  
                        """)
            storyline_output_storypoint_name_list = gr.List(visible=True, type="array", interactive=True, label="Adapt and add Story points, if needed: 📝", 
                                                            scale=1, wrap=True, col_count=[2, "fixed"], elem_id="SPList", headers=["#SP", "Description"])                
            #storyline_output_pretty = gr.Textbox(label="Your Storyline:", lines=13, scale=3, interactive=False)
            submit_button = gr.Button("⚡ Find Slides ⚡", elem_id="visGraph")
            submit_button.click(fn= coordinate_simcalculation, inputs=[storyline_output_storypoint_name_list], outputs=[graphVisual, highest_similarities_gradio_list, queryPlaceholder]
                                ).then(track_user_interaction, inputs=[storyline_output_storypoint_name_list, gr.Textbox("find slides", visible=False)]
                                ).then(js = js_call_draw).then(get_neo4j_response, inputs=[queryPlaceholder], outputs=[responsePlaceholder])



            btn_buildstoryline.click(slide_deck_storyline, 
                                    inputs = [storyline_prompt, nr_storypoints_to_build], 
                                    outputs = [storyline_output_JSON, storyline_output_storypoint_name_list]
                                    )#.then(track_user_interaction, inputs=[storyline_prompt, gr.Textbox("build storyline prompt", visible=False)]
                                    #).then(track_user_interaction, inputs=[storyline_output_storypoint_name_list, gr.Textbox("build storyline gpt output", visible=False)]
                                    #).then(track_user_interaction, inputs=[nr_storypoints_to_build, gr.Textbox("build storyline number of points", visible=False)])
                
            storyline_prompt.submit(slide_deck_storyline, 
                                    inputs = [storyline_prompt, nr_storypoints_to_build], 
                                    outputs = [storyline_output_JSON, storyline_output_storypoint_name_list]
                                    ).then(track_user_interaction, inputs=[storyline_prompt, gr.Textbox("build storyline prompt", visible=False)]
                                    ).then(track_user_interaction, inputs=[storyline_output_storypoint_name_list, gr.Textbox("build storyline gpt output", visible=False)]
                                    ).then(track_user_interaction, inputs=[nr_storypoints_to_build, gr.Textbox("build storyline number of points", visible=False)])

    gr.Markdown("""## 3. Visualize and Filter: 🔍

                Utilize the visualization to align the retrieved slides and storypoints with the objectives and story points defined in Steps 1 and 2:  
                **Filtering the Visualization.**
                Apply filters to better understand the retrieved slides and content that directly correspond to the established learning goals and story points.  
                **Exploring the Graph.**
                Explore relationships within the slide decks, slides, and story points to ensure comprehensive coverage and to identify potential inspiration for your narrative.  
                **Refinements.**
                Should gaps or misalignments be discovered during exploration, revisit Steps 1 and 2 to adjust the learning goals or story points. Then, reapply these refined criteria to filter and explore the visualization again, ensuring the presentation content is tailored and coherent.
                """)


    with gr.Row():
        with gr.Column(scale=2):
            nodeSelector.render()
            filterBTN.render()
            filterBTN.click(fn= construct_hmtl, inputs=[highest_similarities_gradio_list, nodeSelector], outputs=[graphVisual]).then(track_user_interaction, inputs=[nodeSelector, gr.Textbox("filterDropdown", visible=False)]
                                ).then(js = js_call_draw)
            
        with gr.Column(scale=2):
            custom_filtering_output = gr.Textbox(lines=2, scale=3, interactive=True, label = "Describe what you would like to filter for?", placeholder = """For example: 'Filter to only show slides and their respective slide decks that are assigned to the third STORYPOINT'""")
            customfilter_btn = gr.Button("Apply custom filter")
            customfilter_btn.click(custom_filtering, inputs=[custom_filtering_output, queryPlaceholder, responsePlaceholder], outputs=[graphVisual, customFilterQuery]
                                ).then(track_user_interaction, inputs=[custom_filtering_output, gr.Textbox("customFilterPrompt", visible=False)]
                                ).then(track_user_interaction, inputs=[customFilterQuery, gr.Textbox("customFilterQuery", visible=False)]).then(js = js_call_draw)
            
    with gr.Group():
        with gr.Row():
            graphVisual.render()
    #with gr.Row():
        
    

    

gr.close_all()
demo.launch(show_api=False, auth_message = "Hello there! Please log in to access the NarrativeNet Weaver using your Prolific ID as username. The password can be blank.", auth=profile_user)