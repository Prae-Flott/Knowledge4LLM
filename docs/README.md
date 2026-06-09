# Documents and Evaluation Sets

This directory separates knowledge sources from evaluation files.

## Knowledge Sources

The following text files can be used to build the vector knowledge base or the KG-R baseline:

- `robot_knowledges_en.txt`
- `robot_knowledges_de.txt`
- `robot_knowledge_maintenance.txt`
- `iRobot_web_apis.txt`
- `iRobot_web_electric.txt`
- `iRobot_web_mechanic.txt`
- `iRobot_web_overview.txt`
- `iRobot_web_ros2.txt`
- `time_series_features.txt`

The expert-knowledge KG-R baseline uses the maintenance knowledge files. The public-document KG-R variant excludes expert files and uses only the public iRobot documentation.

## Evaluation Sets

The following JSON files are test sets only:

- `test_QA.json`
- `infere_QA.json`
- `data_QA.json`

Do not extract rules, thresholds, or graph edges from these QA files. They are used only to evaluate model and baseline outputs.
