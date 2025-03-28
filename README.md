# NarrativeNet Weaver Web Application

## Overview
The **NarrativeNet Weaver** is an AI-driven web application designed to help users create engaging, structured presentations by leveraging AI-generated storylines, retrieving relevant slide content from a Neo4j database, and visually exploring content connections in a knowledge graph.

## Key Features

- **AI-powered Storyline Generation:** Automatically generates structured story points based on user-defined topics.
- **Neo4j Integration:** Retrieves and visualizes slide decks and story points stored in a Neo4j graph database.
- **Semantic Similarity Matching:** Uses OpenAI embeddings to identify and retrieve the most relevant existing content from the database.
- **Interactive Visualization:** Provides visual exploration of slide decks, slides, and their interconnected story points using Neovis.js.
- **Custom Filtering:** Allows users to refine visualization results through intuitive filters and custom Cypher queries.
- **User Interaction Tracking:** Records user actions and interactions for analytics and continual improvement.


## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Neo4j database
- MySQL database
- OpenAI API key

### Installation
1. Clone the repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export NEO4J_URL=your_neo4j_url
export NEO4J_USERNAME=your_neo4j_username
export NEO4J_PASSWORD=your_neo4j_password
export MYSQLUSER=your_mysql_user
export MYSQLPASSWORD=your_mysql_password
export DBHOST=your_mysql_host
```

4. Launch the app:
```bash
python app.py
```

## Usage
- Enter a topic to generate structured story points.
- Review and adapt generated story points as necessary.
- Find and visualize the most relevant slides from your database.
- Use provided filters or custom queries to refine visualizations.



