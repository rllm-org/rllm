coveragepy_issues = [
    """

Describe the bug
Here below is an item in "files" generated from coverage.py:

"src\textual\widgets\_list_item.py": {
"executed_lines": [
1,
3,
5,
6,
7,
8,
11,
12,
19,
20,
22,
23,
25,
29,
32,
35,
36,
37
],
"summary": {
"covered_lines": 15,
"num_statements": 21,
"percent_covered": 71.42857142857143,
"percent_covered_display": "71",
"missing_lines": 6,
"excluded_lines": 0
},
"missing_lines": [
26,
27,
30,
33,
38,
39
],
"excluded_lines": [],

I would expect len(executed_lines) == summary.covered_lines but the first is 18 and the later is 15. Where does the extra 3 come from?

To Reproduce
I tried this with the following:
pytest . --tb short  --color no --cov=src\textual  --cov-report json --continue-on-collection-errors --disable-warnings
On the https://github.com/Textualize/textual repo.

Expected behavior
My expectation is:
covered_lines + executed_lines + excluded_lines == summary.num_statements, but this seems to not be the case?

""",
]
