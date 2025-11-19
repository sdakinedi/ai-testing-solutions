# """
# Scrum Domain Schema Constants
# Keeps all entity types, relationship types, and KG extraction prompts.
# """

SCRUM_ENTITIES = [
    "Role",
    "Artifact",
    "Event",
    "Practice",
    "Principle",
    "Team",
]

SCRUM_RELATIONS = [
    "FACILITATES",
    "MANAGES",
    "PARTICIPATES_IN",
    "CREATES",
    "USES",
    "OWNS",
    "ATTENDS",
    "RESPONSIBLE_FOR",
]

SCRUM_KG_PROMPT_TEMPLATE = """
You are an expert at extracting entities and relationships from Scrum documentation.

Extract entities of these types: Role, Artifact, Event, Practice, Principle, Team
Extract relationships of these types: 
FACILITATES, MANAGES, PARTICIPATES_IN, CREATES, USES, OWNS, ATTENDS, RESPONSIBLE_FOR

From the following text, identify all entities and their relationships.

Text: {text}

Return a JSON object with entities and relationships.
"""
