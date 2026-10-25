"""Step-by-step Streamlit screen for the document-checking workflow."""
import csv
import io
import os
import streamlit as st
import config
from agent import build_agent, new_run_config, start_run, resume_run

st.set_page_config(page_title='Document Check Agent', layout='wide')
st.title('Document Check Agent')
st.caption('Upload → review rules → approve → check evidence')
st.info('Document text and rules are sent to OpenAI. Use documents you are allowed to share.')

if 'graph' not in st.session_state:
    st.session_state.graph = build_agent()

with st.form('inputs'):
    document = st.file_uploader('Document', type=['pdf', 'docx', 'txt'])
    typed = st.text_area('Rules — one per line')
    rules_file = st.file_uploader('Or upload rules CSV (rule column; optional id)', type=['csv'])
    submitted = st.form_submit_button('Review rulebook')

if submitted:
    if not document or (not typed.strip() and not rules_file):
        st.error('Provide a document and rules.')
    elif typed.strip() and rules_file:
        st.error('Choose typed rules or CSV, so the rulebook is unambiguous.')
    elif document.size > config.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f'Document limit: {config.MAX_UPLOAD_MB} MB.')
    else:
        try:
            # Each upload has an isolated graph, checkpoint store, and index.
            st.session_state.graph = build_agent()
            st.session_state.run_config = new_run_config()
            st.session_state.pop('outcome', None)
            rules_text = rules_file.getvalue().decode('utf-8-sig') if rules_file else typed
            with st.spinner('Reading document and preparing rulebook…'):
                st.session_state.outcome = start_run(
                    st.session_state.graph, st.session_state.run_config,
                    document.getvalue(), document.name, rules_text, rules_file is not None)
        except Exception as exc:
            st.error(f'Could not prepare the run: {exc}')

outcome = st.session_state.get('outcome')
if outcome and outcome.get('__interrupt__'):
    payload = outcome['__interrupt__'][0].value
    st.subheader('Approve your rulebook')
    st.write(payload['question'])
    st.dataframe(payload['rules'], hide_index=True, use_container_width=True)
    approve_col, reject_col = st.columns(2)
    approved = approve_col.button('Approve and check', type='primary')
    rejected = reject_col.button('Reject rulebook')
    if approved or rejected:
        if approved and not os.getenv('OPENAI_API_KEY', '').strip():
            st.error('Add OPENAI_API_KEY to .env and restart Streamlit.')
        else:
            try:
                with st.spinner('Checking each rule (at most two searches per rule)…'):
                    st.session_state.outcome = resume_run(
                        st.session_state.graph, st.session_state.run_config, approved)
                st.rerun()
            except Exception as exc:
                st.error(f'Checking failed ({type(exc).__name__}): {exc}')
elif outcome:
    if not outcome.get('approved'):
        st.info('Rulebook rejected. Edit your rules and start again.')
    else:
        st.subheader('Results')
        st.dataframe(outcome['results'], hide_index=True, use_container_width=True)
        for result in outcome['results']:
            with st.expander(f"{result['id']}: {result['verdict']} — {result['rule']}"):
                st.write(result['explanation'])
                if result['quote']:
                    st.code(result['quote'], language=None)
                    st.caption(result['location'])
                else:
                    st.caption('No verifiable quote found.')
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(outcome['results'][0]))
        writer.writeheader()
        writer.writerows(outcome['results'])
        st.download_button('Download results CSV', output.getvalue(), 'results.csv', 'text/csv')
