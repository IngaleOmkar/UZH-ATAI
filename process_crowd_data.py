from evaluate_crowdsourcing import evaluate_crowd_data
from crowd_clean_malicious import clean_malicious

evaluate_crowd_data('data/crowd_data.tsv', 'data/crowd_majorty_kappa.json')
clean_malicious('data/crowd_data.tsv', 'data/crowd_majorty_kappa.json', 'data/crowd_data_cleaned.tsv')
evaluate_crowd_data('data/crowd_data_cleaned.tsv', 'data/crowd_majorty_kappa_cleaned.json')
