# Zipped shapefiles for Google Earth Engine

Upload each file as a table asset; the asset name is the file name without `.zip`.

- `SUSTD_aoi.zip`: study area (shipped).
- `SUSTD_ordlo.zip`: area occupied before 24 February 2022 (shipped).
- `SUSTD_war_events.zip`: war events of 2022-2024. Not shipped, because the ACLED
  licence does not allow redistribution. `Python_prepare_inputs.py` writes it here
  from your own downloads of ACLED (https://acleddata.com/monitor/ukraine-conflict-monitor)
  and VIINA (https://github.com/zhukovyuri/VIINA); see README, step 1.
