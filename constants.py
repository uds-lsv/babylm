config2size = {
    '1*192': 740160,
    '2*192': 1185024,
    '4*192': 2074752,
    '8*192': 3854208,
    '16*192': 7413120,
    '24*192': 10972032,
    '1*384': 2365056,
    '2*384': 4139520,
    '4*384': 7688448,
    '8*384': 14786304,
    '16*384': 28982016,
    '24*384': 43177728,
    '1*768': 7089408,
    '2*768': 14177280,
    '4*768': 28353024,
    '8*768': 56704512,
    '16*768': 113407488,
    '24*768': 170110464,
    '1*1536': 30693888,
    '2*1536': 59025408,
    '4*1536': 115688448,
    '8*1536': 229014528,
    '16*1536': 455666688,
    '24*1536': 682318848
}

config2size_short = {
    '1*192': "0.74",
    '2*192': "1.19",
    '4*192': "2.07",
    '8*192': "3.85",
    '16*192': "7.41",
    '24*192': "11.0",
    '1*384': "2.37",
    '2*384': "4.14",
    '4*384': "7.69",
    '8*384': "14.8",
    '16*384': "29.0",
    '24*384': "43.2",
    '1*768': "7.10",
    '2*768': "14.2",
    '4*768': "28.4",
    '8*768': "56.7",
    '16*768': "113",
    '24*768': "170",
    '1*1536': "30.7",
    '2*1536': "59.0",
    '4*1536': "116",
    '8*1536': "229",
    '16*1536': "456",
    '24*1536': "682"
}

glue_tasks = ["boolq",  "cola",  "mnli",  "mnli-mm",  "mrpc",  "multirc",  "qnli",  "qqp",  "rte",  "sst2",  "wsc"]

blimp_tasks = ['anaphor_agreement', 'argument_structure', 'binding', 'control_raising', 'determiner_noun_agreement', 'ellipsis', 'filler_gap', 'irregular_forms', 'island_effects', 'npi_licensing', 'quantifiers', 'subject_verb_agreement']

msgs_tasks = ["main_verb_control","control_raising_control","syntactic_category_control","lexical_content_the_control","relative_position_control","main_verb_lexical_content_the","main_verb_relative_token_position","syntactic_category_lexical_content_the","syntactic_category_relative_position","control_raising_lexical_content_the","control_raising_relative_token_position"]

supplement = ["hypernym", "qa_congruence_easy", "qa_congruence_tricky", "subject_aux_inversion", "turn_taking"]

blimp_task2color={
    'anaphor_agreement': "indigo", 
    'argument_structure': "seagreen", 
    'binding': "teal", 
    'control_raising': "goldenrod", 
    'determiner_noun_agreement': "peru", 
    'ellipsis': "fuchsia", 
    'filler_gap': "crimson", 
    'irregular_forms': "grey", 
    'island_effects': "brown", 
    'npi_licensing': "wheat", 
    'quantifiers': "lightgreen", 
    'subject_verb_agreement': "darkcyan",
    "hypernym": "tab:orange", 
    "qa_congruence_easy": "tab:purple", 
    "qa_congruence_tricky": "tab:green", 
    "subject_aux_inversion": "tab:olive", 
    "turn_taking": "tab:blue"
}

glue_task2color={
    'boolq': "indigo", 
    'cola': "seagreen", 
    'mnli': "teal", 
    'mrpc': "goldenrod", 
    'mnli-mm': "peru", 
    'multirc': "fuchsia", 
    'qnli': "crimson", 
    'qqp': "grey", 
    'rte': "brown", 
    'sst2': "wheat", 
    'wsc': "darkcyan", 
}

msgs_task2color={
    "main_verb_control": "indigo", 
    "control_raising_control": "seagreen", 
    "syntactic_category_control": "teal", 
    "control_raising_relative_token_position" : "goldenrod", 
    "lexical_content_the_control": "peru", 
    "relative_position_control": "fuchsia", 
    "main_verb_lexical_content_the": "crimson", 
    "main_verb_relative_token_position": "grey", 
    "syntactic_category_lexical_content_the": "brown", 
    "syntactic_category_relative_position": "wheat", 
    "control_raising_lexical_content_the": "darkcyan", 
}

hs2color = {
    "1536": "b",
    "768": "r",
    "384": "g",
    "192": "m",
}

nlayers2marker = {
    "24": "D",
    "16": "d",
    "8": "v",
    "4": "h",
    "2": "X",
    "1": "*"
}