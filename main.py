from datetime import datetime, timedelta
import json
import logging
import os
import sys
import time

from documentcloud.addon import AddOn
from documentcloud.exceptions import APIError
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError

from ai import get_project_types_from_gpt4, MODEL_NAME
from corrections import corrections

BATCH_RESULTS_JSON_PATH = "data/20241128_batch_results.json"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
documentcloud_logger = logging.getLogger("documentcloud")
documentcloud_logger.setLevel(logging.WARNING)


class AEProjectTypesAddon(AddOn):

    start_time = datetime.now()

    def get_project_id(self):
        """Returns the id of the target project."""

        project = self.data["project"]

        try:
            # if project is an integer, use it as a project ID
            project = int(project)
            return project
        except ValueError:
            # otherwise, get the project id from its title
            # or create it if it does not exist
            project, created = self.client.projects.get_or_create_by_title(project)
            return project.id

    def load_batch_results(self):

        if os.path.isfile(BATCH_RESULTS_JSON_PATH):
            with open(BATCH_RESULTS_JSON_PATH, "r") as batch_results_file:
                batch_results = json.load(batch_results_file)
                logger.info(f"Loaded batch results file {BATCH_RESULTS_JSON_PATH}.")
            return batch_results
        else:
            return {}

    def load_or_create_event_data(self):

        event_data = self.load_event_data()
        if event_data:
            logger.info(
                f"Loaded event_data from DocumentCloud ({len(event_data)} entries)."
            )
            return event_data
        else:
            logger.info("No event_data loaded from DocumentCloud.")
            return {}

        return event_data

    def save_event_data(self):

        self.store_event_data(self.event_data)
        logger.info(
            f"Event data uploaded to DocumentCloud ({len(self.event_data)} entries)"
        )

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")
        filename = f"event_data_project_types_classification_{timestamp}.json"
        with open(filename, "w+") as event_data_file:
            json.dump(self.event_data, event_data_file)
            self.upload_file(event_data_file)
            logger.info(
                f"Event data uploaded to DocumentCloud interface ({len(self.event_data)} entries)"
            )

        with open("event_data.json", "w") as event_data_file:
            json.dump(self.event_data, event_data_file)
        logger.info(
            f"Event data saved to event_data.json ({len(self.event_data)} entries)"
        )

    def set_end_message(self):
        self.processed_count["total"] = (
            self.processed_count["ai"]
            + self.processed_count["ai_batch"]
            + self.processed_count["corrections"]
            + self.processed_count["event_data"]
        )

        end_message = f"Categorized {self.processed_count['total']} documents."
        details = " | ".join(
            [
                f"{x}: {self.processed_count[x]}"
                for x in self.processed_count
                if x != "total"
            ]
        )
        end_message += f" ({details})"

        if self.dry_run:
            end_message += " - (dry run)"

        self.set_message(end_message)
        logger.info(end_message)

    def close_addon(self):
        if not self.dry_run:
            self.save_event_data()
        self.set_end_message()
        sys.exit(0)

    def check_time_limit(self):

        logger.debug("Checking time limit...")
        if self.time_limit != 0:

            limit_seconds = self.time_limit * 60
            now = datetime.now()

            if timedelta.total_seconds(now - self.start_time) > limit_seconds:
                logger.info(f"Closing due to time limit ({self.time_limit} minutes)")
                self.close_addon()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def search_documents(self):
        documents = self.client.documents.search(
            f'+project:{str(self.project)} -data_project_types:* +status:"success" -data_project_types_failed_sources:"{MODEL_NAME}" sort:-data_publication_datetime'
        )
        return documents

    def main(self):

        # Inputs
        self.run_name = self.data.get("run_name", "no name")
        self.time_limit = self.data.get("time_limit", 345)
        self.dry_run = self.data.get("dry_run")

        self.set_message(f"Running... [{self.run_name}]")

        try:
            self.project = self.get_project_id()
        except Exception as e:
            raise Exception("Project error").with_traceback(e.__traceback__)
            sys.exit(1)

        # Event data
        self.event_data = self.load_or_create_event_data()

        # Batch results
        batch_results = self.load_batch_results()

        # Search docs that need to be classified
        logger.info(f"Performing search...")

        try:
            documents = self.search_documents()
            logger.info(f"Found {documents.count} documents to classify.")
        except RetryError:
            logger.error("Failed searching DocumentCloud. Closing...")
            self.close_addon()

        # counts
        self.processed_count = {
            "corrections": 0,
            "ai_batch": 0,
            "event_data": 0,
            "ai": 0,
            "total": 0,
        }

        try:
            # loop
            for doc in documents:

                project_name = doc.description
                source_page_url = doc.data["source_page_url"][0]

                logger.debug(f"Processing doc {doc.title} ({doc.description})")

                # Check corrections first
                if (
                    project_name in corrections
                    and source_page_url in corrections[project_name]
                ):
                    logger.debug("Project found in corrections")

                    if not self.dry_run:
                        logger.debug(
                            f"Matched project types: {corrections[project_name][source_page_url]}"
                        )
                        doc.data["project_types"] = corrections[project_name][
                            source_page_url
                        ]
                        doc.data["project_types_sources"] = ["human"]
                        doc.save()

                        self.processed_count["corrections"] += 1

                # Then batch results
                elif project_name in batch_results:
                    logger.debug("Project name found in batch results.")
                    if not self.dry_run:

                        if batch_results[project_name]:
                            logger.debug(
                                f"Matched project types: {batch_results[project_name]}"
                            )
                            doc.data["project_types"] = batch_results[project_name]
                            doc.data["project_types_sources"] = [MODEL_NAME]
                            doc.save()
                        else:
                            doc.data["project_types_failed_sources"] = [MODEL_NAME]
                            doc.save()
                        self.processed_count["ai_batch"] += 1

                # Then event data
                elif project_name in self.event_data:
                    logger.debug("Project name found in event data.")
                    if not self.dry_run:
                        if self.event_data[project_name][
                            "project_types"
                        ]:  # Project was successfully categorized before
                            logger.debug(
                                f"Matched project types: {self.event_data[project_name]['project_types']}"
                            )
                            doc.data["project_types"] = self.event_data[project_name][
                                "project_types"
                            ]
                            doc.data["project_types_sources"] = self.event_data[
                                project_name
                            ]["project_types_sources"]
                        else:  # Project is in event data but empty, meaning GPT error
                            doc.data["project_types_failed_sources"] = self.event_data[
                                project_name
                            ]["project_types_sources"]
                        doc.save()
                        self.processed_count["event_data"] += 1

                # Get categories from AI
                else:
                    logger.debug("Project to be categorized on the fly by GPT4")
                    if not self.dry_run:
                        project_types_from_ai = get_project_types_from_gpt4(
                            project_name
                        )
                        # Save to event data first
                        # NB: Save regardless of empty response, otherwise docs with same projectcould have different types assigned
                        # if the Add-On fails then succeeds on another doc with the same project name.
                        self.event_data[project_name] = {
                            "project_types": project_types_from_ai,
                            "project_types_sources": [MODEL_NAME],
                        }
                        # Tag document
                        if project_types_from_ai:
                            logger.debug(
                                f"Matched project types: {project_types_from_ai}"
                            )
                            doc.data["project_types"] = project_types_from_ai
                            doc.data["project_types_sources"] = [MODEL_NAME]
                        else:
                            doc.data["project_types_failed_sources"] = [MODEL]
                        doc.save()
                        self.processed_count["ai"] += 1

                self.check_time_limit()
                time.sleep(0.3)

            self.close_addon()

        except APIError:
            logger.warning("DocumentCloud API Error, closing...")
            self.close_addon()


if __name__ == "__main__":
    AEProjectTypesAddon().main()
