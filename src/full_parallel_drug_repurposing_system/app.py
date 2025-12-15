import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid
from threading import Thread
import time
import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Add the parent directory to sys.path to resolve absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from full_parallel_drug_repurposing_system.crew import FullParallelDrugRepurposingSystemCrew
from full_parallel_drug_repurposing_system.models import AgentResponse, MasterResponse
import queue

# --- Load Environment Variables ---
load_dotenv()
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# --- Page Config ---
st.set_page_config(
    page_title="Drug Repurposing AI Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Aesthetics ---
st.markdown("""
<style>
    /* Dark Mode Glassmorphism */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #161b22;
    }
    
    /* Card Styling */
    .css-1r6slb0, .css-1keyCl, .stExpander {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "sessions" not in st.session_state:
    st.session_state.sessions = {} # {session_id: {'messages': [], 'name': 'Timestamp'}}
if "current_session_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_id
    st.session_state.sessions[new_id] = {'messages': [], 'name': 'New Chat', 'research_context': None, 'agent_results': [], 'final_result': None}

# Ensure current session has correct keys
current_sess = st.session_state.sessions[st.session_state.current_session_id]
if 'agent_results' not in current_sess:
    current_sess['agent_results'] = []
if 'final_result' not in current_sess:
    current_sess['final_result'] = None

if "processing_active" not in st.session_state:
    st.session_state.processing_active = False

def get_current_messages():
    return st.session_state.sessions[st.session_state.current_session_id]['messages']

def get_research_context():
    # Construct context dynamically from findings + final report
    session_data = st.session_state.sessions[st.session_state.current_session_id]
    context = "### Intermediate Agent Findings:\n"
    for res in session_data.get('agent_results', []):
         agent_name = res.get('agent', 'Unknown Agent')
         summary = res.get('data', {}).get('summary', res.get('full_raw', ''))
         context += f"**Agent {agent_name}**:\n{summary}\n\n"
    
    if session_data.get('final_result'):
         context += f"\n### Final Strategic Report:\n{session_data['final_result']}\n"
         
    return context

def add_message(role, content, visualization=None):
    st.session_state.sessions[st.session_state.current_session_id]['messages'].append({
        "role": role,
        "content": content,
        "visualization": visualization
    })

def format_agent_name(name):
    return name.replace("_", " ").title()

# --- Sidebar: Session History (Left Column) ---
with st.sidebar:
    st.title("🗂️ History")
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.sessions[new_id] = {'messages': [], 'name': 'New Chat', 'research_context': None, 'agent_results': [], 'final_result': None}
        st.rerun()
    
    st.markdown("---")
    for session_id, session_data in list(st.session_state.sessions.items())[::-1]:
        if st.button(f"🗓️ {session_data['name']}", key=session_id, use_container_width=True):
            st.session_state.current_session_id = session_id
            st.rerun()

# --- Main Layout ---
st.title("🧬 AI Drug Repurposing Agent")
st.markdown("### Discover new therapeutic uses for existing drugs via parallel agentic research.")

# Split Layout: Process (Middle) vs Chat (Right)
col_process, col_chat = st.columns([0.65, 0.35], gap="large")

# --- Middle Column: Agent Processing & Results ---
with col_process:
    st.subheader("Research Process")
    
    # Input Area
    if prompt := st.chat_input("Enter a drug name (e.g., Aspirin, Metformin)"):
        # User Message
        st.session_state.sessions[st.session_state.current_session_id]['agent_results'] = [] # Clear previous results
        st.session_state.sessions[st.session_state.current_session_id]['final_result'] = None
        
        # Update Session Name if New
        if st.session_state.sessions[st.session_state.current_session_id]['name'] == 'New Chat':
             st.session_state.sessions[st.session_state.current_session_id]['name'] = f"Research: {prompt}"
        
        st.session_state.processing_active = True
        
        # Start Background Thread
        def task_callback(task_output):
            # Put the task execution result into the queue
            data = None
            if hasattr(task_output, 'pydantic') and task_output.pydantic:
                 try:
                    data = task_output.pydantic.model_dump()
                 except:
                    pass
            
            return {
                "type": "task",
                "agent": task_output.agent,
                "summary": task_output.raw[:200] + "..." if len(task_output.raw) > 200 else task_output.raw,
                "data": data,
                "full_raw": task_output.raw
            }

        def run_crew(crew_inputs, result_queue, msg_queue):
            try:
                # Local wrapper to capture callbacks
                def local_callback(task_output):
                    msg = task_callback(task_output)
                    msg_queue.put(msg)

                crew_instance = FullParallelDrugRepurposingSystemCrew().crew()
                
                # Assign callbacks to tasks
                for task in crew_instance.tasks:
                    task.callback = local_callback
                    
                result = crew_instance.kickoff(inputs=crew_inputs)
                result_queue.put({"type": "result", "data": result})
            except Exception as e:
                result_queue.put({"type": "error", "error": str(e)})

        # Initialize Queues
        if 'msg_queue' not in st.session_state:
            st.session_state.msg_queue = queue.Queue()
        if 'result_queue' not in st.session_state:
            st.session_state.result_queue = queue.Queue()

        inputs = {'compound_name': prompt}
        thread = Thread(target=run_crew, args=(inputs, st.session_state.result_queue, st.session_state.msg_queue))
        thread.start()
        
        # We need to rerun to start the polling loop in the main flow
        st.rerun()

    # --- Polling and Rendering Loop ---
    
    # Display Existing Results (Persistent)
    current_results = st.session_state.sessions[st.session_state.current_session_id]['agent_results']
    for res in current_results:
        agent_display_name = format_agent_name(res['agent'])
        with st.expander(f"✅ {agent_display_name} Finished", expanded=False):
            data = res.get('data')
            if data:
                if 'summary' in data:
                    st.markdown(f"**Summary:** {data['summary']}")
                if 'key_findings' in data:
                    st.markdown("**Key Findings:**")
                    for insight in data['key_findings']:
                        st.markdown(f"- {insight['summary']} (Confidence: {insight.get('confidence', 0.0)})")
                if 'next_steps' in data and data['next_steps']:
                    st.markdown("**Next Steps:**")
                    for step in data['next_steps']:
                        st.markdown(f"- {step}")
            else:
                 st.markdown(res.get('full_raw', res.get('summary')))

    # Check for new messages if active
    if st.session_state.processing_active:
        with st.status("🔍 Agents Working...", expanded=True) as status:
            if 'msg_queue' in st.session_state:
                try:
                    while not st.session_state.msg_queue.empty():
                        msg = st.session_state.msg_queue.get_nowait()
                        if msg["type"] == "task":
                            st.session_state.sessions[st.session_state.current_session_id]['agent_results'].append(msg)
                            st.rerun() # Rerun to render it in the persistent list above
                except queue.Empty:
                    pass
                
                # Check for completion
                if 'result_queue' in st.session_state and not st.session_state.result_queue.empty():
                    res = st.session_state.result_queue.get()
                    st.session_state.processing_active = False
                    
                    if res["type"] == "error":
                        st.error(res["error"])
                    elif res["type"] == "result":
                        status.update(label="✅ Research Complete!", state="complete", expanded=False)
                        
                        # Process Final Result
                        final_result = None
                        if hasattr(res["data"], 'pydantic') and res["data"].pydantic:
                            final_result = res["data"].pydantic
                        else:
                            final_result = res["data"].raw
                        
                        st.session_state.sessions[st.session_state.current_session_id]['final_result'] = final_result
                        st.rerun()
            
            time.sleep(1) 
            st.rerun()

    # Display Final Result
    final_res = st.session_state.sessions[st.session_state.current_session_id]['final_result']
    if final_res:
         st.markdown("### 📋 Final Strategic Report")
         
         # Fallback / Duck Typing check for finding data properties
         is_structured = False
         if hasattr(final_res, 'executive_summary') and hasattr(final_res, 'key_insights'):
             is_structured = True
         
         if is_structured:
            st.markdown(f"#### Executive Summary: {final_res.drug_name}")
            st.info(final_res.executive_summary)
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Overall Confidence", f"{final_res.overall_confidence * 100:.1f}%")
            with m_col2:
                st.metric("Success Probability", f"{final_res.success_probability * 100:.1f}%")
            
            st.markdown("#### Key Insights")
            for insight in final_res.key_insights:
                st.markdown(f"- **{insight.summary}**: {insight.details} (*Source: {insight.source}*)")
         else:
            # Clean string representation attempt
            try:
                st.text(str(final_res)) 
            except:
                st.markdown("Result available in context but has display error.")


# --- Right Column: Chat Interface ---
with col_chat:
    st.subheader("💬 Analysis Chat")
    st.caption("Interact with the agents' findings here.")
    
    # Message Display
    chat_container = st.container(height=600)
    with chat_container:
        for msg in get_current_messages():
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("visualization"):
                    st.plotly_chart(msg["visualization"], use_container_width=True)

    # Chat Input
    if q_prompt := st.chat_input("Ask a question about the findings..."):
        # Add to history
        add_message("user", q_prompt)
        st.rerun()

    # Handle Chat Response (after rerun)
    messages = get_current_messages()
    if messages and messages[-1]['role'] == 'user':
        context = get_research_context()
        if context:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Gemini Support
                            if "GEMINI_API_KEY" not in os.environ:
                                st.error("GEMINI_API_KEY not found in .env file. Please add it.")
                            else:
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                prompt_content = f"You are a helpful pharmaceutical research assistant.\n\nContext from Research Agents:\n{context}\n\nUser Question: {messages[-1]['content']}\n\nAnswer:"
                                
                                response = model.generate_content(prompt_content)
                                answer = response.text
                                
                                st.markdown(answer)
                                add_message("assistant", answer)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error executing chat: {str(e)}")
        else:
             with chat_container:
                with st.chat_message("assistant"):
                     resp = "I don't have any research results yet. Please start a research task in the left panel first."
                     st.markdown(resp)
                     add_message("assistant", resp)
