import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid
from threading import Thread
import time
from full_parallel_drug_repurposing_system.crew import FullParallelDrugRepurposingSystemCrew

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
    .css-1r6slb0, .css-1keyCl {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
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
    st.session_state.sessions[new_id] = {'messages': [], 'name': 'New Chat'}

def get_current_messages():
    return st.session_state.sessions[st.session_state.current_session_id]['messages']

def add_message(role, content, visualization=None):
    st.session_state.sessions[st.session_state.current_session_id]['messages'].append({
        "role": role,
        "content": content,
        "visualization": visualization
    })

# --- Sidebar: Session History ---
with st.sidebar:
    st.title("🗂️ History")
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.sessions[new_id] = {'messages': [], 'name': 'New Chat'}
        st.rerun()
    
    st.markdown("---")
    for session_id, session_data in list(st.session_state.sessions.items())[::-1]:
        if st.button(f"🗓️ {session_data['name']}", key=session_id, use_container_width=True):
            st.session_state.current_session_id = session_id
            st.rerun()

# --- Main Interface ---
st.title("🧬 AI Drug Repurposing Agent")
st.markdown("### Discover new therapeutic uses for existing drugs via parallel agentic research.")

# Display Chat History
for msg in get_current_messages():
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("visualization"):
            st.plotly_chart(msg["visualization"], use_container_width=True)

# --- Agent Processing Logic ---
def extract_json_section(text):
    """Extracts JSON block from the text"""
    try:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return json.loads(text[start:end])
        elif "Structured Data Section" in text:
            # Fallback for simpler extraction if marked
            pass
    except:
        pass
    return None

def create_visualizations(data):
    """Creates Plotly charts from structured data"""
    figs = []
    if not data:
        return figs
    
    # Example: Bar Chart
    if "metrics" in data:
         # Generic handler if 'metrics' key exists
         pass
    
    # Assuming data is list of opportunities with scores
    if isinstance(data, list):
         df = pd.DataFrame(data)
         if not df.empty and "Success Probability" in df.columns:
             fig = px.bar(df, x="Opportunity", y="Success Probability", 
                          title="Success Probability of Repurposing Candidates",
                          color="Success Probability",
                          color_continuous_scale="Viridis")
             figs.append(fig)
    return figs

# Chat Input
if prompt := st.chat_input("Enter a drug name (e.g., Aspirin, Metformin)"):
    # User Message
    st.chat_message("user").markdown(prompt)
    add_message("user", prompt)
    
    # Update Session Name if New
    if st.session_state.sessions[st.session_state.current_session_id]['name'] == 'New Chat':
         st.session_state.sessions[st.session_state.current_session_id]['name'] = f"Research: {prompt}"

    # Agent Execution
    if prompt:
        with st.chat_message("assistant"):
            st.markdown("Initialising **Drug Repurposing Crew**...")
            
            # Container for real-time updates
            status_container = st.status("🔍 Agents Working...", expanded=True)
            output_placeholder = status_container.empty()
            
            # Queue for inter-thread communication
            import queue
            msg_queue = queue.Queue()
            
            def task_callback(task_output):
                # Put the task execution result into the queue
                msg_queue.put({
                    "type": "task",
                    "agent": task_output.agent,
                    "summary": task_output.raw[:200] + "..." if len(task_output.raw) > 200 else task_output.raw
                })

            def run_crew(crew_inputs, result_queue):
                try:
                    # Create crew with callback
                    # Note: We need to patch the tasks to add the callback since they are created by decorator
                    crew_instance = FullParallelDrugRepurposingSystemCrew().crew()
                    
                    # Monkey-patch callbacks into tasks? 
                    # Easier: The Crew object allows 'task_callback' or 'step_callback' in newer versions?
                    # Documentation says Crew(..., task_callback=...)
                    # FullParallelDrugRepurposingSystemCrew returns a Crew object in .crew() method.
                    # We can iterate over crew_instance.tasks and assign the callback
                    for task in crew_instance.tasks:
                        task.callback = task_callback
                        
                    result = crew_instance.kickoff(inputs=crew_inputs)
                    result_queue.put({"type": "result", "data": result})
                except Exception as e:
                    result_queue.put({"type": "error", "error": str(e)})

            # Start Crew in Background Thread
            inputs = {'compound_name': prompt}
            result_queue = queue.Queue()
            thread = Thread(target=run_crew, args=(inputs, result_queue))
            thread.start()
            
            # Polling Loop
            logs = []
            final_result = None
            
            while thread.is_alive() or not msg_queue.empty():
                try:
                    # Non-blocking get
                    msg = msg_queue.get_nowait()
                    if msg["type"] == "task":
                        log_entry = f"✅ **{msg['agent']}**: {msg['summary']}"
                        logs.append(log_entry)
                        # Construct log string
                        log_str = "\n\n".join(logs)
                        output_placeholder.markdown(log_str)
                except queue.Empty:
                    time.sleep(0.1)
                
            # Wait for final result
            thread.join()
            
            # Check for result or error
            if not result_queue.empty():
                res = result_queue.get()
                if res["type"] == "error":
                    status_container.update(label="❌ Error Occurred", state="error")
                    st.error(res["error"])
                elif res["type"] == "result":
                    status_container.update(label="✅ Research Complete!", state="complete", expanded=False)
                    final_result = res["data"].raw
            
            # Render Final Output
            if final_result:
                st.markdown("### 📋 Final Strategic Report")
                st.markdown(final_result)
                
                # Visualization
                viz_data = extract_json_section(final_result)
                final_vis = None
                if viz_data:
                    figs = create_visualizations(viz_data)
                    for fig in figs:
                        st.plotly_chart(fig)
                        final_vis = fig
                
                add_message("assistant", final_result, visualization=final_vis)
