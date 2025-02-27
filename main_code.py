import streamlit as st
import json
import tempfile
from typing import List, Dict, Any, TypedDict
from datetime import datetime, timedelta
from groq import Client
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64
import re
import pytz
from langgraph.graph import Graph, StateGraph
import os
import traceback

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Personal Assistant",
    page_icon="🤖",
    layout="wide"
)

# Define state management
class AgentState(TypedDict):
    query: str
    actions: List[str]
    current_agent: str
    final_response: str

# Initialize Groq client inside a function instead of at module level
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

def get_google_credentials():
    """Get Google credentials from Streamlit secrets or local file."""
    if "google_credentials" in st.secrets:
        # Create credentials from secrets
        creds_info = {k: v for k, v in st.secrets["google_credentials"].items()}
        
        # Save to temporary file for functions expecting a file path
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp.write(json.dumps(creds_info).encode())
        temp.close()
        return temp.name
    else:
        return 'token.json'  # Fallback for local development

def safe_json_parse(text, default=None):
    """Safely parse JSON with fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to fix common JSON formatting issues
        if '"agents":' in text:
            # Extract just the agents array if possible
            try:
                match = re.search(r'"agents":\s*(\[[^\]]+\])', text)
                if match:
                    agents_json = match.group(1)
                    return {"agents": json.loads(agents_json)}
            except Exception:
                pass
        return default if default is not None else {"agents": ["calendar_list"]}

def create_calendar_event(query: str) -> str:
    try:
        credentials_path = get_google_credentials()
        
        calendar = build('calendar', 'v3', 
            credentials=Credentials.from_authorized_user_file(credentials_path, 
                ['https://www.googleapis.com/auth/calendar.events']))
        
        timezone = 'Asia/Kolkata'
        now = datetime.now(pytz.timezone(timezone))
        
        # Enhanced prompt to explicitly handle time ranges
        prompt = f"""Current time: {now.strftime('%Y-%m-%d %I:%M %p')} IST
        Parse this event: "{query}"
        IMPORTANT: Pay careful attention to time ranges. If a time range like "from X to Y" is specified, make sure to capture both the start and end times correctly.
        
        Return JSON with:
        - summary: event title (e.g. "Meeting with Om")
        - date: the date in YYYY-MM-DD format. For relative dates like "tomorrow", calculate based on current date.
        - start_time: start time in HH:MM format (24-hour)
        - end_time: end time in HH:MM format (24-hour), default to 1 hour after start time if not specified
        - location: location of the event (e.g. "Innovation Hub", "Conference Room 3"), or empty string if not specified
        - attendees: list of people attending the event (names or roles), or empty array if not specified
        - description: any additional details about the event, or empty string if not specified
        
        For example, if the query is "Schedule a project kickoff meeting on 22nd March from 4 PM to 6 PM at the Innovation Hub", 
        the end_time should be "18:00" (not "17:00")."""

        groq_client = get_groq_client()
        if not groq_client:
            return "❌ Error initializing Groq client. Please check your API key."
            
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        
        # Parse the LLM response
        try:
            event_details = json.loads(response_content)
        except json.JSONDecodeError as e:
            return f"❌ Failed to parse event details: {e}"
        
        # Parse start time
        start_datetime_str = f"{event_details['date']}T{event_details['start_time']}:00"
        start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%dT%H:%M:%S')
        
        # Parse end time - with improved handling
        # Ensure end_time exists and is properly formatted
        if 'end_time' in event_details and event_details['end_time'] and event_details['end_time'] != event_details['start_time']:
            end_datetime_str = f"{event_details['date']}T{event_details['end_time']}:00"
            try:
                end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%dT%H:%M:%S')
                # Ensure end time is after start time
                if end_datetime <= start_datetime:
                    end_datetime = start_datetime + timedelta(hours=1)
            except ValueError:
                end_datetime = start_datetime + timedelta(hours=1)
        else:
            # Default to 1 hour if end_time is missing or invalid
            end_datetime = start_datetime + timedelta(hours=1)
            
        # Directly parse from input if LLM fails
        if "from" in query.lower() and "to" in query.lower() and "pm" in query.lower():
            try:
                # Extract time range directly from the query as a backup
                from_pattern = r'from\s+(\d+(?:\:\d+)?)\s*(?:am|pm)\s+to\s+(\d+(?:\:\d+)?)\s*(?:am|pm)'
                time_match = re.search(from_pattern, query.lower(), re.IGNORECASE)
                
                if time_match:
                    start_hr = int(time_match.group(1))
                    end_hr = int(time_match.group(2))
                    
                    # Convert to 24-hour format if PM
                    if "pm" in query.lower():
                        if start_hr < 12:
                            start_hr += 12
                        if end_hr < 12:
                            end_hr += 12
                    
                    # Recreate the end time
                    end_datetime = start_datetime.replace(hour=end_hr, minute=0)
            except Exception:
                pass
        
        tz = pytz.timezone(timezone)
        start_datetime = tz.localize(start_datetime)
        end_datetime = tz.localize(end_datetime)
        
        # Create event dictionary
        event = {
            'summary': event_details['summary'],
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': timezone
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': timezone
            }
        }
        
        # Add location if specified
        if 'location' in event_details and event_details['location']:
            event['location'] = event_details['location']
        
        # Add description with attendees if specified
        description = ""
        if 'description' in event_details and event_details['description']:
            description = event_details['description']
        
        # Handle attendees
        attendees = []
        if 'attendees' in event_details and event_details['attendees']:
            if isinstance(event_details['attendees'], list):
                attendees = event_details['attendees']
            else:
                attendees = [event_details['attendees']]
        
        if attendees:
            attendees_list = ", ".join([str(a) for a in attendees if a is not None])
            description += f"\n\nAttendees: {attendees_list}"
        
        if description:
            event['description'] = description
        
        # Insert event into calendar
        result = calendar.events().insert(calendarId='primary', body=event).execute()
        
        # Prepare response
        response_parts = [
            f"✅ Event Scheduled Successfully!",
            f"Event: {event_details['summary']}",
            f"Date: {start_datetime.strftime('%A, %B %d, %Y')}",
            f"Time: {start_datetime.strftime('%I:%M %p')} to {end_datetime.strftime('%I:%M %p')}"
        ]
        
        if 'location' in event_details and event_details['location']:
            response_parts.append(f"Location: {event_details['location']}")
        
        if attendees:
            attendees_str = ", ".join([str(a) for a in attendees if a is not None])
            if attendees_str:
                response_parts.append(f"Attendees: {attendees_str}")
        
        response_parts.append(f"View in Calendar: {result.get('htmlLink')}")
        
        return "\n".join(response_parts)
    except Exception as e:
        return f"❌ Failed to create calendar event: {str(e)}"

def list_calendar_events(query: str) -> str:
    try:
        credentials_path = get_google_credentials()
        
        calendar = build('calendar', 'v3', 
            credentials=Credentials.from_authorized_user_file(credentials_path, 
                ['https://www.googleapis.com/auth/calendar.readonly']))
        
        prompt = f"""From '{query}' extract number of events to show. Default is 10. Return just the number."""
        
        groq_client = get_groq_client()
        if not groq_client:
            return "❌ Error initializing Groq client. Please check your API key."
            
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_content = response.choices[0].message.content.strip()
        
        try:
            num_events = int(response_content)
        except:
            num_events = 10
            
        # Handle special case for small numbers from the query
        if "my 2 upcoming events" in query.lower() or "list my 2 upcoming events" in query.lower():
            num_events = 2
        
        events_result = calendar.events().list(
            calendarId='primary',
            timeMin=datetime.utcnow().isoformat() + 'Z',
            timeMax=(datetime.utcnow() + timedelta(days=30)).isoformat() + 'Z',
            maxResults=num_events,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        output = [f"\nUpcoming {num_events} Events:", "-" * 30]
        ist = pytz.timezone('Asia/Kolkata')
        
        for event in events:
            if 'dateTime' in event['start']:
                start = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00')).astimezone(ist)
                end = datetime.fromisoformat(event['end']['dateTime'].replace('Z', '+00:00')).astimezone(ist)
                output.append(f"\nTime: {start.strftime('%b %d, %I:%M %p')} - {end.strftime('%I:%M %p')}")
            else:
                output.append(f"\nDate: {datetime.fromisoformat(event['start']['date']).strftime('%b %d')} (All day)")
            
            output.append(f"Event: {event.get('summary', 'No Title')}")
            if 'location' in event:
                output.append(f"Location: {event['location']}")
            output.append("-" * 30)
        
        if not events:
            output.append("\nNo upcoming events found.")
            
        return "\n".join(output)
    except Exception as e:
        return f"❌ Failed to list calendar events: {str(e)}"

# Gmail Agent Tool
def send_email(query: str) -> str:
    try:
        credentials_path = get_google_credentials()
        
        # Initialize Gmail service
        creds = Credentials.from_authorized_user_file(credentials_path, 
            ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
        gmail = build('gmail', 'v1', credentials=creds)
        
        # Extract email address
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', query)
        if not email_match:
            return "❌ No email address found in the query!"
            
        recipient = email_match.group()
        
        # Generate email content
        prompt = f"""
        Based on this request: "{query}"
        Generate a professional email with:
        1. Subject line
        2. Professional greeting
        3. Main message
        4. Professional closing
        5. Signature: Best regards,\nMilind Warade

        Format:
        SUBJECT: [subject line]
        [email body with signature]
        """
        
        groq_client = get_groq_client()
        if not groq_client:
            return "❌ Error initializing Groq client. Please check your API key."
            
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content.strip()
        
        subject = content.split('\n')[0].replace('SUBJECT:', '').strip()
        body = '\n'.join(content.split('\n')[1:]).strip()
        
        try:
            sender = gmail.users().getProfile(userId='me').execute()['emailAddress']
            
            message = MIMEText(body)
            message['to'] = recipient
            message['from'] = sender
            message['subject'] = subject
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
            
            final_response = f"""✅ Email sent successfully!
To: {recipient}
Subject: {subject}
Message:
{body}"""
            return final_response
        except Exception as e:
            return f"❌ Email sending failed: {e}"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"

# Agent Manager Functions
def route_query(state: AgentState) -> AgentState:
    query = state["query"]
    
    # Direct pattern matching for common cases to avoid LLM calls when possible
    if any(phrase in query.lower() for phrase in ["schedule", "create event", "add to calendar", "book"]):
        state["actions"] = ["calendar_create"]
        return state
        
    if any(phrase in query.lower() for phrase in ["list", "show", "upcoming events", "my events", "what events"]):
        state["actions"] = ["calendar_list"]
        return state
        
    if any(phrase in query.lower() for phrase in ["send email", "email to", "write email", "compose email"]):
        state["actions"] = ["email"]
        return state
    
    # Fallback to LLM for more complex queries
    prompt = f"""Analyze this query: "{query}"
    Determine which agent(s) should handle it:
    - calendar_create: Create calendar event (e.g. "schedule a meeting", "create event")
    - calendar_list: List calendar events (e.g. "show my events", "list meetings")
    - email: Send email (e.g. "send email", "compose message")
    
    Return JSON with array of required agents.
    Example outputs:
    - "schedule a meeting tomorrow" -> {"agents": ["calendar_create"]}
    - "send email to john@example.com" -> {"agents": ["email"]}
    - "show my next 5 events" -> {"agents": ["calendar_list"]}"""
    
    try:
        groq_client = get_groq_client()
        
        if not groq_client:
            state["actions"] = ["calendar_list"]  # Default fallback
            state["final_response"] = "❌ Error initializing Groq client. Please check your API key."
            return state
            
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        
        # Safer JSON parsing with fallback
        parsed_json = safe_json_parse(response_content, default={"agents": ["calendar_list"]})
        required_agents = parsed_json.get("agents", ["calendar_list"])
        
        state["actions"] = required_agents
        return state
    except Exception as e:
        # Log the exception
        print(f"Error in route_query: {str(e)}")
        # Ensure it's a list of strings
        state["actions"] = ["calendar_list"]  # Default fallback
        state["final_response"] = f"❌ Error determining required agents: {str(e)}"
        return state

def execute_tools(state: AgentState) -> AgentState:
    responses = []
    
    if not state["actions"]:
        state["final_response"] = "No specific actions were identified from your request. Please try again with a clearer request."
        return state
    
    for action in state["actions"]:
        try:
            if action == "calendar_create":
                response = create_calendar_event(state["query"])
                responses.append(response)
            elif action == "calendar_list":
                response = list_calendar_events(state["query"])
                responses.append(response)
            elif action == "email":
                response = send_email(state["query"])
                responses.append(response)
            else:
                responses.append(f"Unknown action: {action}")
        except Exception as e:
            responses.append(f"❌ Error executing {action}: {str(e)}")
    
    if not responses:
        state["final_response"] = "I couldn't process your request. Please try again."
    else:
        state["final_response"] = "\n\n".join(responses)
    
    return state

def agent_manager(query: str) -> str:
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("route", route_query)
    workflow.add_node("execute", execute_tools)
    
    # Add edges
    workflow.add_edge("route", "execute")
    workflow.set_entry_point("route")
    
    # Compile workflow
    graph = workflow.compile()
    
    try:
        # Execute workflow
        result = graph.invoke({
            "query": query,
            "actions": [],
            "current_agent": "",
            "final_response": ""
        })
        
        if "final_response" in result and result["final_response"]:
            return result["final_response"]
        else:
            return "Sorry, I couldn't generate a response for your request."
    except Exception as e:
        traceback.print_exc()  # Print full traceback for debugging
        return f"Error processing your request: {str(e)}"

# Add custom CSS for styling input box
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        min-height: 100px;
    }
    
    .stTextArea > div > div > textarea {
        min-height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤖 AI Personal Assistant")
st.markdown("""
This assistant can help you with:
- 📅 Creating calendar events
- 📋 Listing upcoming calendar events
- 📧 Sending emails
""")

# Input section
query = st.text_area(
    "What can I help you with?",
    placeholder="Examples:\n- Schedule a meeting with John tomorrow at 3pm\n- Show my next 5 calendar events\n- Send an email to example@email.com about project updates",
    height=100
)

# Process button
if st.button("Process Request", type="primary"):
    if query:
        with st.spinner("Processing your request..."):
            try:
                # Get response from agent manager
                response = agent_manager(query)

                # Ensure response is a string
                if not isinstance(response, str):
                    response = "❌ Error: No valid response received."

                # Display response
                st.markdown("### Response:")
                
                # Use Streamlit container
                with st.container():
                    # Split response into lines and format them
                    lines = response.split('\n')
                    for line in lines:
                        if "Event Scheduled Successfully" in line or line.startswith("✅"):
                            st.success(line)
                        elif "Email sent successfully" in line:
                            st.success(line)
                        elif "Upcoming" in line:
                            st.subheader(line)
                        elif line.startswith("❌"):
                            st.error(line)
                        elif line.startswith("-" * 10):  # Separator lines
                            st.markdown("---")
                        else:
                            st.write(line)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                traceback.print_exc()  # Print full traceback for debugging
    else:
        st.warning("Please enter a request first.")

# Footer
st.markdown("---")
st.markdown("Made by Milind Warade")

# For local execution
if __name__ == "__main__":
    pass