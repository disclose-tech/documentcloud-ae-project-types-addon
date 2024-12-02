import os
import json
import logging

from openai import OpenAI
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError


MODEL_NAME = "gpt-4o"

PROJECT_TYPES_CSV_PATH = "data/project_types.csv"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

client = OpenAI(api_key=os.environ["TOKEN"])


def make_project_types_string():
    """Get the CSV file containing project types in raw text format."""
    with open(PROJECT_TYPES_CSV_PATH, "r") as types_file:
        project_types_string = types_file.read()

    return project_types_string


def get_project_types_list():
    """Get a list of possible project types."""
    project_types_list = list(pd.read_csv(PROJECT_TYPES_CSV_PATH)["Catégorie"])

    return project_types_list


def make_schema():
    """Prepare schema for custom response format."""
    project_types_list = get_project_types_list()

    schema = {
        "name": "categories_response",
        "schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "description": "Liste de catégories pertinentes pour le projet considéré.",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"enum": project_types_list},
                },
            },
            "required": ["categories"],
            "additionalProperties": False,
            "strict": True,
        },
    }

    response_json_schema = {
        "type": "json_schema",
        "json_schema": schema,
    }

    return response_json_schema


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(project_name):

    response_json_schema = make_schema()

    project_types_string = make_project_types_string()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Tu es un super assistant spécialisé dans les projets soumis à autorisation environnementale en France",
            },
            {
                "role": "user",
                "content": f"""
            Voici un projet ou un programme dont le nom est : "{project_name}".
            Voici maintenant un tableau au format CSV comporte des catégories et des exemples de mots-clés associés : 
            {project_types_string}
            En te basant sur cette catégorisation et le nom du projet, trouve la ou les catégories (minimum 1, maximum 3) qui correspondent le mieux au nom du projet.

            Avant de répondre, vérifie que ta réponse est conforme aux 6 consignes suivantes, ou modifie ta réponse pour qu'elle le soit.
                1. Les catégories "PLU", "PLUi", "SCOT", "ZAC", "CC" et "PCAET", doivent être utilsées de manière très restrictive, uniquement si le nom du projet contient le nom de la catégorie ou un de ses mots-clés.
                2. Les catégories "PLU", "PLUi" "SCOT", "ZAC" ou "CC" doivent être utilisées seules, sauf si le nom du projet mentionne un second projet connexe.
                3. Les catégories "Aménagements urbains" et "Aménagement ruraux" sont à utiliser en dernier recours et sans autre catégorie.
                4. Un projet ne peut pas être à la fois dans "Aménagements urbains" et "Aménagements ruraux", il faut choisir l'une des 2.
                5. Un projet ne peut pas être à la fois dans "Cours d'eau" et dans "Travaux maritimes", il faut choisir l'une des 2. 
                6. La catégorie "Hydroélectricité" ne doit pas s'appliquer à des barrages dont il n'est pas précisé qu'ils soient électriques.
            """,
            },
        ],
        max_tokens=150,
        temperature=0,
        response_format=response_json_schema,
    )

    result = response.choices[0].message.content.strip()

    return result


def get_project_types_from_gpt4(project_name):
    """Get the categories matching the project name from OpenAI"""

    try:
        result = get_completion(project_name)
    except RetryError:
        logger.error(f"Error: failed to get a response from the OpenAI API.")
        result = "{'categories': []}"

    try:
        result_dict = json.loads(result)
        project_types_result = result_dict["categories"]
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON for project name: {project}")
        project_types_result = []

    # Double checking project_types
    project_types_list = get_project_types_list()
    project_types_result = [x for x in project_types_result if x in project_types_list]

    project_types_result = sorted(project_types_result)

    return project_types_result
