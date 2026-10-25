"""Run the real agent and compare verdicts by rule ID against a CSV key."""
import argparse
import csv
from pathlib import Path
from agent import build_agent, new_run_config, start_run, resume_run
from tools import load_rules

BASE_DIR = Path(__file__).resolve().parent


def default_test_sets():
    return [
        ('Smoke test', BASE_DIR / 'examples/document.txt', BASE_DIR / 'examples/rules.csv',
         BASE_DIR / 'examples/answers.csv'),
        ('PulseCare technical file', BASE_DIR / 'test_data/PulseCare_Technical_File.docx',
         BASE_DIR / 'test_data/rules_technical_file.csv',
         BASE_DIR / 'test_data/answer_key_technical_file.csv'),
        ('PulseCare test report', BASE_DIR / 'test_data/PulseCare_Test_Report.docx',
         BASE_DIR / 'test_data/rules_test_report.csv',
         BASE_DIR / 'test_data/answer_key_test_report.csv'),
    ]


def load_answer_key(path, rules):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        verdict_column = 'verdict' if 'verdict' in fields else 'expected'
        if 'id' not in fields or verdict_column not in fields:
            raise ValueError('Answer key CSV needs id and verdict (or expected) columns.')
        answers = {}
        for row in reader:
            rule_id = row['id'].strip()
            verdict = row[verdict_column].strip().upper()
            if not rule_id or rule_id in answers:
                raise ValueError('Answer key IDs must be nonempty and unique.')
            if verdict not in {'MET', 'NOT MET', 'UNCLEAR'}:
                raise ValueError(f'Invalid verdict for {rule_id}: {verdict}')
            answers[rule_id] = verdict
    if set(answers) != {r['id'] for r in rules}:
        raise ValueError('Answer key IDs must match the rules CSV exactly.')
    return answers


def compare_results(results, answers):
    actual = {r['id']: r for r in results}
    wrong = []
    for rule_id, expected in answers.items():
        result = actual.get(rule_id, {})
        if result.get('verdict') != expected:
            wrong.append({'id': rule_id, 'expected': expected,
                          'actual': result.get('verdict', 'MISSING'),
                          'rule': result.get('rule', ''),
                          'reason': result.get('reason', 'No verdict returned.')})
    return len(answers) - len(wrong), wrong


def evaluate_set(name, document, rules_path, answers_path, approve=False):
    print(f'\n{name}', flush=True)
    document = Path(document)
    rules_text = Path(rules_path).read_text(encoding='utf-8-sig')
    rules = load_rules(rules_text, csv_mode=True)
    answers = load_answer_key(answers_path, rules)
    graph, run_config = build_agent(), new_run_config()
    paused = start_run(graph, run_config, document.read_bytes(), document.name, rules_text, True)
    for rule in paused['__interrupt__'][0].value['rules']:
        print(f"{rule['id']}: {rule['rule']}")
    approved = approve or input('Approve this rulebook? [y/N] ').strip().lower() == 'y'
    if not approved:
        resume_run(graph, run_config, False)
        print('Evaluation cancelled.')
        return None
    print('Checking document…', flush=True)
    outcome = resume_run(graph, run_config, True)
    for result in outcome['results']:
        print(f"{result['id']}: {result['verdict']} — {result['reason']}")
    correct, wrong = compare_results(outcome['results'], answers)
    print(f'{name}: {correct}/{len(answers)} correct', flush=True)
    for item in wrong:
        print(f"- {item['id']}: expected {item['expected']}, got {item['actual']} — "
              f"{item['rule']} — Reason: {item['reason']}")
    if not wrong:
        print('No incorrect rules.')
    return correct, len(answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document')
    parser.add_argument('--rules')
    parser.add_argument('--answers')
    parser.add_argument('--approve', action='store_true',
                        help='Explicitly approve the displayed rulebooks for automated evaluation.')
    args = parser.parse_args()
    custom = (args.document, args.rules, args.answers)
    if any(custom) and not all(custom):
        parser.error('--document, --rules, and --answers must be supplied together.')
    sets = [('Custom test', *custom)] if all(custom) else default_test_sets()
    total_correct = total_rules = 0
    failed = []
    for name, document, rules_path, answers_path in sets:
        try:
            score = evaluate_set(name, document, rules_path, answers_path, args.approve)
            if score is None:
                return
            total_correct += score[0]
            total_rules += score[1]
        except Exception as exc:
            print(f'{name}: ERROR ({type(exc).__name__}): {exc}', flush=True)
            failed.append(name)
    label = 'Total (completed sets)' if failed else 'Total'
    print(f'\n{label}: {total_correct}/{total_rules} correct')
    if failed:
        parser.exit(2, f"Failed sets: {', '.join(failed)}\n")


if __name__ == '__main__':
    main()
