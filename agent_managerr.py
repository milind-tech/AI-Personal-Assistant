import json
import streamlit as st
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
api_key = os.getenv("API_KEY")  # Load from environment


# # Debug API Key Loading
# st.write("Checking Streamlit Secrets...")  # This helps verify execution
# api_key = st.secrets.get("GROQ_API_KEY", None)

# if not api_key:
#     st.error("🚨 GROQ_API_KEY is missing! Please check Streamlit secrets.")
# else:
#     st.success("✅ GROQ_API_KEY is loaded!")

# Initialize Groq Client
if api_key:
    # groq_client = Groq(api_key=api_key)

    groq_client = (api_key=api_key)  # Ensure correct import


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
# Initialize Groq client
groq_client = Client(api_key=st.secrets["GROQ_API_KEY"])

# Define state management
class AgentState(TypedDict):
    query: str
    actions: List[str]
    current_agent: str
    final_response: str



def create_calendar_event(query: str) -> str:
    calendar = build('calendar', 'v3', 
        credentials=Credentials.from_authorized_user_file(get_google_credentials(), 
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

    response = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    # Parse the LLM response
    event_details = json.loads(response.choices[0].message.content)
    
    # Debug step - print the parsed JSON to help diagnose issues
    print(f"Parsed event details: {json.dumps(event_details, indent=2)}")
    
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
                print("Warning: End time is not after start time, defaulting to 1 hour duration")
                end_datetime = start_datetime + timedelta(hours=1)
        except ValueError as e:
            print(f"Error parsing end time: {e}, defaulting to 1 hour duration")
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
                print(f"Direct parsing from query: end time set to {end_hr}:00")
        except Exception as e:
            print(f"Error in direct parsing: {e}")
    
    # Add explicit debug log
    print(f"Final times: Start = {start_datetime}, End = {end_datetime}")
    
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
        f"Event Scheduled Successfully! 🎉",
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


def list_calendar_events(query: str) -> str:
    calendar = build('calendar', 'v3', 
        credentials=Credentials.from_authorized_user_file(get_google_credentials(), 
            ['https://www.googleapis.com/auth/calendar.readonly']))
    
    prompt = f"""From '{query}' extract number of events to show. Default is 10. Return just the number."""
    response = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        num_events = int(response.choices[0].message.content.strip())
    except:
        num_events = 10
    
    events = calendar.events().list(
        calendarId='primary',
        timeMin=datetime.utcnow().isoformat() + 'Z',
        timeMax=(datetime.utcnow() + timedelta(days=30)).isoformat() + 'Z',
        maxResults=num_events,
        singleEvents=True,
        orderBy='startTime'
    ).execute().get('items', [])
    
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
    
    return "\n".join(output)

# Gmail Agent Tool
def send_email(query: str) -> str:
    # Initialize Gmail service
    creds = Credentials.from_authorized_user_file(get_google_credentials(), 
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
        gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
        
        return f"""✅ Email sent successfully!
To: {recipient}
Subject: {subject}
Message:
{body}"""
    except Exception as e:
        return f"❌ Email sending failed: {e}"

# Agent Manager Functions
def route_query(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    
    prompt = f"""Analyze this query: "{query}"
    Determine which agent(s) should handle it:
    - calendar_create: Create calendar event (e.g. "schedule a meeting", "create event")
    - calendar_list: List calendar events (e.g. "show my events", "list meetings")
    - email: Send email (e.g. "send email", "compose message")
    
    Return JSON with array of required agents.
    Example outputs:
    - "schedule a meeting tomorrow" -> ["calendar_create"]
    - "send email to john@example.com" -> ["email"]
    - "show my next 5 events" -> ["calendar_list"]"""
    
    response = groq_client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    required_agents = json.loads(response.choices[0].message.content)["agents"]
    state["actions"] = required_agents
    return state

def execute_tools(state: AgentState) -> Dict[str, Any]:
    responses = []
    
    for action in state["actions"]:
        if action == "calendar_create":
            responses.append(create_calendar_event(state["query"]))
        elif action == "calendar_list":
            responses.append(list_calendar_events(state["query"]))
        elif action == "email":
            responses.append(send_email(state["query"]))
    
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
    
    # Execute workflow
    result = graph.invoke({
        "query": query,
        "actions": [],
        "current_agent": "",
        "final_response": ""
    })
    
    return result["final_response"]

if __name__ == "__main__":
    # Simple command line interface for testing
    query = input("Enter your request: ")
    print("\nProcessing your request...")
    response = agent_manager(query)
    print("\nResponse:", response)