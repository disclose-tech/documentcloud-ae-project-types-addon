
# DocumentCloud Add-On

[Please see the Add-On documentation](https://github.com/MuckRock/documentcloud-hello-world-addon/wiki/)

# DocumentCloud AE Project Types Addon

DocumentCloud Add-On to categorize AE documents by project types.

## Event data

Event data lists the projects already categorized by GPT, as `{project_name: {"project_types": [...], "project_types_sources": [...]}}` (empty `project_types` means GPT failed). It is split in two:

- **Archive** ([event_data/archive.json](event_data/archive.json)): committed in the repo and loaded from disk at each run.
- **Live event data**: stored on DocumentCloud, holding only the projects categorized since the last archive refresh. It is saved every 25 GPT calls and when the run ends (only if it changed).

To refresh the archive (when the live event data grows, e.g. yearly), run the following, then commit `event_data/archive.json`:

```bash
DC_USERNAME=... DC_PASSWORD=... python scripts/refresh_event_data_archive.py --event-id <scheduled event ID>
```

The following run removes the archived projects from the live event data.
