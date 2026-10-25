"""A bounded, inspectable LangGraph workflow with human approval."""
from typing import TypedDict
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from retrieval import HybridSearch
from tools import read_document, load_rules, search_evidence, judge_rule, ask_user, is_prohibition
import config


class State(TypedDict, total=False):
    document: bytes
    filename: str
    rules_text: str
    csv_mode: bool
    chunks: list
    rules: list
    approved: bool
    position: int
    attempts: int
    results: list
    current: dict
    evidence: list


def build_agent(model=None, index_factory=HybridSearch):
    # Keep the search index in this session, outside serialized checkpoints.
    indexes = {}

    def read(state):
        return {'chunks': read_document(state['document'], state['filename'])}

    def rules(state):
        return {'rules': load_rules(state['rules_text'], state.get('csv_mode', False)),
                'position': 0, 'attempts': 0, 'results': []}

    def approve(state):
        answer = ask_user('Approve this rulebook before checking the document?', state['rules'])
        return {'approved': answer is True}

    def search(state, config: RunnableConfig):
        thread = config['configurable']['thread_id']
        if thread not in indexes:
            indexes[thread] = index_factory(state['chunks'])
        rule = state['rules'][state['position']]
        query = rule['rule']
        if state['attempts']:
            alternate = state['current'].get('alternate_query', '').strip()
            if not alternate or alternate.casefold() == query.casefold():
                alternate = f'Explicit supporting or contradictory statements, conditions and exceptions: {query}'
            query = alternate
        found = search_evidence(indexes[thread], query)
        # A document-wide prohibition needs all text to check for violations.
        if is_prohibition(rule):
            found = state['chunks']
        # Keep evidence from both attempts so the second verdict sees both.
        previous = state.get('evidence', []) if state['attempts'] else []
        merged = {c.metadata['chunk_id']: c for c in previous + found}
        return {'evidence': list(merged.values()), 'attempts': state['attempts'] + 1}

    def judge(state):
        llm = model or ChatOpenAI(model=config.CHAT_MODEL, temperature=config.TEMPERATURE,
                                 max_retries=config.MAX_RETRIES, timeout=config.REQUEST_TIMEOUT)
        result = judge_rule(llm, state['rules'][state['position']], state['evidence'])
        # The retry judges merged evidence and supplies the final judgment.
        return {'current': result}

    def after_judgment(state):
        if not state['current']['strong'] and state['attempts'] < config.MAX_SEARCH_ATTEMPTS:
            return 'search_evidence'
        return 'save_result'

    def save(state):
        result = {k: v for k, v in state['current'].items()
                  if k not in ('strong', 'alternate_query', 'no_evidence')}
        # After the retry, use its absence finding over the merged evidence.
        if state['current']['no_evidence'] and state['attempts'] >= config.MAX_SEARCH_ATTEMPTS:
            result.update(quote='', location='')  # Keep the rule-specific absence verdict.
        result['search_attempts'] = state['attempts']
        return {'results': state['results'] + [result],
                'position': state['position'] + 1, 'attempts': 0, 'evidence': []}

    def cleanup(state, config: RunnableConfig):
        indexes.pop(config['configurable']['thread_id'], None)
        return {}

    graph = StateGraph(State)
    for name, node in [('read_document', read), ('load_rules', rules), ('ask_user', approve),
                       ('search_evidence', search), ('judge_rule', judge),
                       ('save_result', save), ('cleanup', cleanup)]:
        graph.add_node(name, node)
    graph.add_edge(START, 'read_document')
    graph.add_edge('read_document', 'load_rules')
    graph.add_edge('load_rules', 'ask_user')
    graph.add_conditional_edges('ask_user', lambda s: 'search_evidence' if s['approved'] else 'cleanup')
    graph.add_edge('search_evidence', 'judge_rule')
    graph.add_conditional_edges('judge_rule', after_judgment)
    graph.add_conditional_edges('save_result', lambda s:
        'search_evidence' if s['position'] < len(s['rules']) else 'cleanup')
    graph.add_edge('cleanup', END)
    return graph.compile(checkpointer=InMemorySaver())


def new_run_config():
    return {'configurable': {'thread_id': str(uuid4())},
            'recursion_limit': config.MAX_GRAPH_STEPS}


def start_run(graph, run_config, data, filename, rules_text, csv_mode=False):
    return graph.invoke({'document': data, 'filename': filename,
                         'rules_text': rules_text, 'csv_mode': csv_mode}, run_config)


def resume_run(graph, run_config, approved):
    return graph.invoke(Command(resume=approved), run_config)
