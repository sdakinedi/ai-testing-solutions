"""
Domain constants for Scrum Knowledge Graph
Configured to extract entities and relationships
"""

from neo4j_graphrag.experimental.pipeline.kg_builder import OnError


# ENTITY DEFINITIONS


SCRUM_ENTITIES = [
    "Person",          # Scrum Master, Product Owner, Development Team Member
    "Role",            # Scrum Master, Product Owner, Development Team
    "Artifact",        # Product Backlog, Sprint Backlog, Increment
    "Event",           # Sprint, Sprint Planning, Daily Scrum, Sprint Review, Sprint Retrospective
    "Concept",        # Scrum, Agile, User Story, Definition of Done
    "Practice",        # Backlog Refinement, Estimation, Velocity
    "Organization",    # Scrum Team
]


# RELATIONSHIP DEFINITIONS

SCRUM_RELATIONS = [
    "HAS_ROLE",
    "MANAGES",
    "PARTICIPATES_IN",
    "CREATES",
    "MAINTAINS",
    "ATTENDS",
    "FACILITATES",
    "OWNS",
    "RESPONSIBLE_FOR",
    "CONTAINS",
    "PRODUCES",
    "PART_OF",
    "RELATED_TO",
    "OCCURS_DURING",
    "FOLLOWS",
]


# PROMPT TEMPLATE FOR ENTITY EXTRACTION
SCRUM_KG_PROMPT_TEMPLATE = '''
You are an expert in Scrum methodology and knowledge graph construction.
Your task is to extract structured information from the provided text and create a knowledge graph.

Instructions:
1. Identify all entities related to Scrum from the categories listed below
2. Extract relationships between entities using the types provided below
3. Be specific: like extract "Scrum Master", "Product Owner", "Sprint Planning" as distinct entities
4. Create clear, meaningful relationships between entities
5. Preserve important context and details

Entity Extraction Guidelines:
- Roles: Extract specific role names (e.g., "Scrum Master", "Product Owner", "Development Team")
- Events: Extract Scrum ceremonies (e.g., "Sprint Planning", "Daily Scrum", "Sprint Review")
- Artifacts: Extract Scrum artifacts (e.g., "Product Backlog", "Sprint Backlog", "Increment")
- Practices: Extract Scrum practices (e.g., "Backlog Refinement", "Sprint Goal")

Relationship Guidelines:
- Connect roles to their responsibilities
- Link events to their participants and outcomes
- Associate artifacts with the roles that manage them
- Show the flow and dependencies between Scrum events

**Text to analyze:**
{text}

**Output Format:**
Return a list of nodes and relationships in JSON format:
{{
    "nodes": [
        {{"id": "unique_id", "label": "EntityType", "properties": {{"name": "Entity Name", "description": "..."}}}}
    ],
    "relationships": [
        {{"type": "RELATIONSHIP_TYPE", "start_node_id": "id1", "end_node_id": "id2"}}
    ]
}}
'''


# PIPELINE CONFIGURATION
KG_BUILDER_CONFIG = {
    "on_error": OnError.IGNORE,  # Continue on errors
    "perform_entity_resolution": True,  # Merge similar entities
}