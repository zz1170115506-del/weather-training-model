CURRENT: seventh_run_course_only.json
Run from the repository root:
  python code/run_task1.py check --config configs/task1/seventh_run_course_only.json --data-dir <COURSE_DATA_DIR>
  python code/run_task1.py train --config configs/task1/seventh_run_course_only.json --data-dir <COURSE_DATA_DIR>

Other JSON files are historical recipes retained to reproduce recorded
course-only experiments and algorithm ablations. They are never automatically
run by the current recipe. In particular, fourth_run_search.json and
sixth_run_course_only.json enable expensive historical nested searches.
No public-source recipe is supported. Original course CSV hashes are pinned.

For a compact verification copy, use code/verify_task1_release.py. It includes
only the selected current config, its runtime and dependencies; historical
configs and results remain in this repository, outside that exported package.
