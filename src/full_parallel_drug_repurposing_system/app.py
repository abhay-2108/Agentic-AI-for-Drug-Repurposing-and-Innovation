import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid
from threading import Thread
import time
from full_parallel_drug_repurposing_system.crew import FullParallelDrugRepurposingSystemCrew
from full_parallel_drug_repurposing_system.models import AgentResponse, MasterResponse
from crewai import LLM

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
    st.session_state.current_session_id = new_id
    st.session_state.sessions[new_id] = {'messages': [], 'name': 'New Chat', 'research_context': None}

def get_current_messages():
    return st.session_state.sessions[st.session_state.current_session_id]['messages']

def get_research_context():
    return st.session_state.sessions[st.session_state.current_session_id].get('research_context')

def set_research_context(context):
    st.session_state.sessions[st.session_state.current_session_id]['research_context'] = context

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
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.sessions[new_id] = {'messages': [], 'name': 'New Chat', 'research_context': None}
        st.rerun()
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
                data = None
                if hasattr(task_output, 'pydantic') and task_output.pydantic:
                     try:
                        data = task_output.pydantic.model_dump()
                     except:
                        pass
                
                msg_queue.put({
                    "type": "task",
                    "agent": task_output.agent,
                    "summary": task_output.raw[:200] + "..." if len(task_output.raw) > 200 else task_output.raw,
                    "data": data
                })

            def run_crew(crew_inputs, result_queue):
                try:
                    # Create crew with callback
                    crew_instance = FullParallelDrugRepurposingSystemCrew().crew()
                    
                    # Assign callbacks to tasks
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
                        agent_name = msg['agent']
                        summary = msg['summary']
                        data = msg.get('data')
                        
                        if data:
                            # Render structured agent output in the status container
                            with status_container:
                                with st.expander(f"✅ {agent_name} Finished", expanded=False):
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
                            # Fallback to raw log
                            log_entry = f"✅ **{agent_name}**: {summary}"
                            logs.append(log_entry)
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
                    # Extract pydantic object from final result
                    if hasattr(res["data"], 'pydantic') and res["data"].pydantic:
                        final_result = res["data"].pydantic
                    else:
                        final_result = res["data"].raw
            
            # Render Final Output
            if final_result:
                st.markdown("### 📋 Final Strategic Report")
                
                final_content_str = ""
                
                if isinstance(final_result, MasterResponse):
                    # Structured Final Display
                    st.markdown(f"#### Executive Summary: {final_result.drug_name}")
                    st.info(final_result.executive_summary)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Overall Confidence", f"{final_result.overall_confidence * 100:.1f}%")
                    with col2:
                        st.metric("Success Probability", f"{final_result.success_probability * 100:.1f}%")
                    
                    st.markdown("#### Key Insights")
                    for insight in final_result.key_insights:
                        st.markdown(f"- **{insight.summary}**: {insight.details} (*Source: {insight.source}*)")
                        
                    st.markdown("#### Strategic Recommendations")
                    for rec in final_result.recommendations:
                        priority_color = "red" if rec.priority.lower() == "high" else "orange" if rec.priority.lower() == "medium" else "green"
                        st.markdown(f"- :{priority_color}[**{rec.priority}**] **{rec.action}**: {rec.rationale}")
                        
                    final_content_str = final_result.model_dump_json(indent=2)
                else:
                    # Fallback to raw text
                    st.markdown(final_result)
                    final_content_str = str(final_result)
                
                # Visualization (Generic if possible, or skip if fully structured)
                # If we want to keep the old viz logic, we can try to extract data from the structured object
                final_vis = None
                
                add_message("assistant", final_content_str, visualization=final_vis)
                set_research_context(final_content_str)

# --- Post-Research Chat ---
context = get_research_context()
if context:
    st.markdown("---")
    st.header("💬 Chat with Research Findings")
    st.caption("Ask questions about the generated report.")
    
    # Initialize chat history for Q&A if not exists (we can reuse main history or separate)
    # Using main history for continuity
    
    if q_prompt := st.chat_input("Ask a follow-up question...", key="qa_input"):
        st.chat_message("user").markdown(q_prompt)
        add_message("user", q_prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Simple LLM call using CrewAI's LLM wrapper or LiteLLM
                    llm = LLM(model="ollama/minimax-m2:cloud", base_url="http://localhost:11434")
                    
                    # Construct prompt with context
                    messages = [
                        {"role": "system", "content": "You are a helpful pharmaceutical research assistant. Answer the user's question based strictly on the provided research context below."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q_prompt}"}
                    ]
                    
                    response = llm.call(messages)
                    st.markdown(response)
                    add_message("assistant", response)
                except Exception as e:
                    st.error(f"Error executing chat: {str(e)}")
