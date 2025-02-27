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
import traceback

# Set up page configuration
st.set_page_config(page_title="AI Personal Assistant", page_icon="🤖")

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
        print("Debug: Groq API key is available")
        return Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        print(f"Debug: Error initializing Groq client: {e}")
        return None

def get_google_credentials():
    """Get Google credentials from Streamlit secrets or local file."""
    if "google_credentials" in st.secrets:
        print("Debug: Google credentials found in Streamlit secrets")
        # Create credentials from secrets
        creds_info = {k: v for k, v in st.secrets["google_credentials"].items()}
        
        # Save to temporary file for functions expecting a file path
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp.write(json.dumps(creds_info).encode())
        temp.close()
        print(f"Debug: Credentials saved to temporary file: {temp.name}")
        return temp.name
    else:
        print("Debug: Using local token.json for credentials")
        return 'token.json'  # Fallback for local development

def safe_json_parse(text, default=None):
    """Safely parse JSON with fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Debug: JSON parsing error: {e}, text: {text}")
        # Try to fix common JSON formatting issues
        if '"agents":' in text:
            # Extract just the agents array if possible
            try:
                import re
                match = re.search(r'"agents":\s*(\[[^\]]+\])', text)
                if match:
                    agents_json = match.group(1)
                    return {"agents": json.loads(agents_json)}
            except Exception:
                pass
        return default if default is not None else {"agents": ["calendar_list"]}

def create_calendar_event(query: str) -> str:
    print(f"Debug: create_calendar_event called with query: {query}")
    try:
        credentials_path = get_google_credentials()
        print(f"Debug: Using credentials from: {credentials_path}")
        
        calendar = build('calendar', 'v3', 
            credentials=Credentials.from_authorized_user_file(credentials_path, 
                ['https://www.googleapis.com/auth/calendar.events']))
        print("Debug: Calendar service built successfully")
        
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
            error_msg = "❌ Error initializing Groq client. Please check your API key."
            print(f"Debug: {error_msg}")
            return error_msg
            
        print("Debug: Sending request to Groq API for event parsing")
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        print(f"Debug: Received response from Groq API: {response_content}")
        
        # Parse the LLM response
        try:
            event_details = json.loads(response_content)
            print(f"Debug: Parsed event details: {json.dumps(event_details, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"Debug: Error parsing event JSON: {e}")
            return f"❌ Failed to parse event details: {e}"
        
        # Parse start time
        start_datetime_str = f"{event_details['date']}T{event_details['start_time']}:00"
        print(f"Debug: Start datetime string: {start_datetime_str}")
        start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%dT%H:%M:%S')
        
        # Parse end time - with improved handling
        # Ensure end_time exists and is properly formatted
        if 'end_time' in event_details and event_details['end_time'] and event_details['end_time'] != event_details['start_time']:
            end_datetime_str = f"{event_details['date']}T{event_details['end_time']}:00"
            try:
                print(f"Debug: End datetime string: {end_datetime_str}")
                end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%dT%H:%M:%S')
                # Ensure end time is after start time
                if end_datetime <= start_datetime:
                    print("Debug: Warning: End time is not after start time, defaulting to 1 hour duration")
                    end_datetime = start_datetime + timedelta(hours=1)
            except ValueError as e:
                print(f"Debug: Error parsing end time: {e}, defaulting to 1 hour duration")
                end_datetime = start_datetime + timedelta(hours=1)
        else:
            # Default to 1 hour if end_time is missing or invalid
            print("Debug: Using default 1 hour duration")
            end_datetime = start_datetime + timedelta(hours=1)
            
        # Directly parse from input if LLM fails
        if "from" in query.lower() and "to" in query.lower() and "pm" in query.lower():
            try:
                print("Debug: Attempting direct time range parsing from query")
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
                    print(f"Debug: Direct parsing from query: end time set to {end_hr}:00")
            except Exception as e:
                print(f"Debug: Error in direct parsing: {e}")
        
        # Add explicit debug log
        print(f"Debug: Final times: Start = {start_datetime}, End = {end_datetime}")
        
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
        print("Debug: Inserting event into calendar")
        result = calendar.events().insert(calendarId='primary', body=event).execute()
        print(f"Debug: Event created successfully with ID: {result.get('id')}")
        
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
        
        final_response = "\n".join(response_parts)
        print(f"Debug: Final response: {final_response}")
        return final_response
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Debug: Error in create_calendar_event: {str(e)}")
        print(f"Debug: Traceback: {error_trace}")
        return f"❌ Failed to create calendar event: {str(e)}"

def list_calendar_events(query: str) -> str:
    print(f"Debug: list_calendar_events called with query: {query}")
    try:
        credentials_path = get_google_credentials()
        print(f"Debug: Using credentials from: {credentials_path}")
        
        calendar = build('calendar', 'v3', 
            credentials=Credentials.from_authorized_user_file(credentials_path, 
                ['https://www.googleapis.com/auth/calendar.readonly']))
        print("Debug: Calendar service built successfully")
        
        prompt = f"""From '{query}' extract number of events to show. Default is 10. Return just the number."""
        
        groq_client = get_groq_client()
        if not groq_client:
            error_msg = "❌ Error initializing Groq client. Please check your API key."
            print(f"Debug: {error_msg}")
            return error_msg
            
        print("Debug: Sending request to Groq API for event count parsing")
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_content = response.choices[0].message.content.strip()
        print(f"Debug: Received response from Groq API: {response_content}")
        
        try:
            num_events = int(response_content)
            print(f"Debug: Parsed number of events: {num_events}")
        except:
            print("Debug: Could not parse number of events, using default of 10")
            num_events = 10
        
        print(f"Debug: Fetching {num_events} calendar events")
        events_result = calendar.events().list(
            calendarId='primary',
            timeMin=datetime.utcnow().isoformat() + 'Z',
            timeMax=(datetime.utcnow() + timedelta(days=30)).isoformat() + 'Z',
            maxResults=num_events,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        print(f"Debug: Received {len(events)} events from calendar API")
        
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
            
        final_response = "\n".join(output)
        print(f"Debug: Final response: {final_response}")
        return final_response
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Debug: Error in list_calendar_events: {str(e)}")
        print(f"Debug: Traceback: {error_trace}")
        return f"❌ Failed to list calendar events: {str(e)}"

# Gmail Agent Tool
def send_email(query: str) -> str:
    """Send an email based on the user query with improved content generation and error handling."""
    try:
        credentials_path = get_google_credentials()
        if not credentials_path:
            return "❌ Google credentials not available. Unable to send email."
        
        # Import Google libraries only when needed
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from email.mime.text import MIMEText
        import base64
        
        # Initialize Gmail service
        creds = Credentials.from_authorized_user_file(credentials_path, 
            ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
        gmail = build('gmail', 'v1', credentials=creds)
        
        # Extract email address with improved regex that handles multiple formats
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        email_match = re.search(email_pattern, query)
        
        if not email_match:
            # Try to find keywords like "to" followed by potential recipients
            to_match = re.search(r'(?:to|send to|email to)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
            if to_match:
                recipient_name = to_match.group(1).strip()
                return f"❌ Could not find a valid email address for '{recipient_name}'. Please include a complete email address."
            return "❌ No email address found in the query! Please include a valid email address."
            
        recipient = email_match.group()
        
        # Extract subject with improved pattern matching
        subject = "No Subject"
        subject_patterns = [
            r'(?:subject|about|regarding|re)[:|\s]\s*([^\.]+)',
            r'(?:email|message|send)\s+(?:about|regarding)\s+([^\.]+)'
        ]
        
        for pattern in subject_patterns:
            subject_match = re.search(pattern, query, re.IGNORECASE)
            if subject_match:
                subject = subject_match.group(1).strip()
                break
                
        # Extract content/topic information
        content_info = query.replace(f"to {recipient}", "").replace(f"about {subject}", "")
        content_info = re.sub(r'send\s+(?:an)?\s*email', '', content_info, flags=re.IGNORECASE).strip()
        
        # Generate email content
        groq_client = get_groq_client()
        if groq_client:
            # Use LLM for email content with more specific instructions
            prompt = f"""
            Generate a professional email based on this request: "{query}"
            
            From the request, I understand:
            - Recipient: {recipient}
            - Subject: {subject}
            - Content relates to: {content_info}
            
            Create a concise, professional email that:
            1. Has an appropriate greeting
            2. Clearly communicates the main message in 2-3 paragraphs
            3. Includes a professional closing
            4. Signature: Best regards,\\nMilind Warade
            
            Format your response as the complete email body only, ready to send.
            """
            
            try:
                response = groq_client.chat.completions.create(
                    model="gemma2-9b-it",  # Using a consistent model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,  # Adding some creativity while remaining professional
                    max_tokens=500  # Limiting length for efficiency
                )
                
                body = response.choices[0].message.content.strip()
                
                # Fallback if LLM returns empty or very short content
                if len(body) < 30:
                    raise Exception("Generated content too short")
            except Exception as e:
                # Fallback to template-based email
                body = generate_template_email(recipient, subject, content_info)
        else:
            # Use template-based email generation when LLM is unavailable
            body = generate_template_email(recipient, subject, content_info)
        
        try:
            # Get sender email
            sender = gmail.users().getProfile(userId='me').execute()['emailAddress']
            
            # Create and send message
            message = MIMEText(body)
            message['to'] = recipient
            message['from'] = sender
            message['subject'] = subject
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
            
            # Format response with email details
            final_response = f"""✅ Email sent successfully!
To: {recipient}
Subject: {subject}
Message:
{body}"""
            return final_response
        except Exception as e:
            return f"❌ Email sending failed: {str(e)}"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}\n{traceback.format_exc()}"

def generate_template_email(recipient, subject, content_info):
    """Generate a template-based email when LLM is unavailable."""
    # Extract name from email if possible
    name_match = re.match(r'([^@]+)@', recipient)
    recipient_name = name_match.group(1).replace(".", " ").title() if name_match else "Recipient"
    
    # Current date for the email
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Generate a more contextual email based on available information
    if "list" in content_info.lower() or "top" in content_info.lower():
        return f"""Hello {recipient_name},

I hope this email finds you well. I am writing to share the information you requested about {subject or 'the requested topic'}.

{content_info.strip() or 'As discussed, I wanted to follow up with the relevant details. Please let me know if you need any clarification or have additional questions.'}

Thank you for your attention to this matter.

Best regards,
Milind Warade

Date: {current_date}"""
    else:
        return f"""Hello {recipient_name},

I hope this email finds you well. I am reaching out regarding {subject or 'an important matter'}.

{content_info.strip() or 'I wanted to touch base with you about our recent discussions. Please review the information and let me know your thoughts or if you need any additional details.'}

Thank you for your time and consideration.

Best regards,
Milind Warade

Date: {current_date}"""

# Agent Manager Functions
def route_query(state: AgentState) -> Dict[str, Any]:
    print(f"Debug: route_query started with query: {state['query']}")
    query = state["query"]
    
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
        print("Debug: Groq client initialized")
        
        if not groq_client:
            print("Debug: Groq client initialization failed")
            state["actions"] = []
            state["final_response"] = "❌ Error initializing Groq client. Please check your API key."
            return state
            
        print("Debug: Sending request to Groq API for route determination")
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        print(f"Debug: Received response from Groq API: {response_content}")
        
        # Safer JSON parsing with fallback
        parsed_json = safe_json_parse(response_content, default={"agents": ["calendar_list"]})
        required_agents = parsed_json.get("agents", ["calendar_list"])
        
        print(f"Debug: Required agents: {required_agents}")
        state["actions"] = required_agents
        return state
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Debug: Error in route_query: {str(e)}")
        print(f"Debug: Traceback: {error_trace}")
        state["actions"] = ["calendar_list"]  # Default fallback
        state["final_response"] = f"❌ Error determining required agents: {str(e)}"
        return state

def execute_tools(state: AgentState) -> Dict[str, Any]:
    print(f"Debug: execute_tools started with actions: {state['actions']}")
    responses = []
    
    if not state["actions"]:
        print("Debug: No actions to execute")
        state["final_response"] = "No specific actions were identified from your request. Please try again with a clearer request."
        return state
    
    for action in state["actions"]:
        print(f"Debug: Executing action: {action}")
        try:
            if action == "calendar_create":
                response = create_calendar_event(state["query"])
                print(f"Debug: calendar_create response length: {len(response)}")
                responses.append(response)
            elif action == "calendar_list":
                response = list_calendar_events(state["query"])
                print(f"Debug: calendar_list response length: {len(response)}")
                responses.append(response)
            elif action == "email":
                response = send_email(state["query"])
                print(f"Debug: email response length: {len(response)}")
                responses.append(response)
            else:
                print(f"Debug: Unknown action: {action}")
                responses.append(f"Unknown action: {action}")
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Debug: Error executing {action}: {str(e)}")
            print(f"Debug: Traceback: {error_trace}")
            responses.append(f"❌ Error executing {action}: {str(e)}")
    
    if not responses:
        print("Debug: No responses generated")
        state["final_response"] = "I couldn't process your request. Please try again."
    else:
        print(f"Debug: Number of responses: {len(responses)}")
        state["final_response"] = "\n\n".join(responses)
    
    return state

def agent_manager(query: str) -> str:
    print(f"Debug: Agent manager started with query: {query}")
    
    # Create workflow
    workflow = StateGraph(AgentState)
    print("Debug: StateGraph created")
    
    # Add nodes
    workflow.add_node("route", route_query)
    workflow.add_node("execute", execute_tools)
    print("Debug: Nodes added to workflow")
    
    # Add edges
    workflow.add_edge("route", "execute")
    workflow.set_entry_point("route")
    print("Debug: Edges configured and entry point set")
    
    # Compile workflow
    graph = workflow.compile()
    print("Debug: Workflow compiled")
    
    try:
        # Execute workflow
        print("Debug: About to invoke the graph")
        result = graph.invoke({
            "query": query,
            "actions": [],
            "current_agent": "",
            "final_response": ""
        })
        print(f"Debug: Graph execution complete. Result keys: {list(result.keys())}")
        
        if "final_response" in result and result["final_response"]:
            print(f"Debug: Final response length: {len(result['final_response'])}")
            return result["final_response"]
        else:
            print("Debug: No final response in result")
            return "Sorry, I couldn't generate a response for your request."
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Debug: Error in graph execution: {str(e)}")
        print(f"Debug: Traceback: {error_trace}")
        return f"Error processing your request: {str(e)}"

def direct_test(query: str, forced_agent: str) -> str:
    """For testing: directly call a specific agent function."""
    print(f"Debug: direct_test with agent {forced_agent} and query {query}")
    try:
        if forced_agent == "calendar_create":
            return create_calendar_event(query)
        elif forced_agent == "calendar_list":
            return list_calendar_events(query)
        elif forced_agent == "email":
            return send_email(query)
        else:
            return f"Unknown agent: {forced_agent}"
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Debug: Error in direct_test: {str(e)}")
        print(f"Debug: Traceback: {error_trace}")
        return f"Error: {str(e)}"

def agent_manager_with_fallback(query: str) -> str:
    print(f"Debug: agent_manager_with_fallback started with query: {query}")
    try:
        response = agent_manager(query)
        print(f"Debug: agent_manager returned response of length: {len(response) if response else 0}")
        
        # If response is empty, provide a fallback
        if not response or response.strip() == "":
            print("Debug: Empty response detected, using fallback")
            return "I couldn't process your request properly. Please try again or try a different request."
        
        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Debug: Error in agent_manager_with_fallback: {str(e)}")
        print(f"Debug: Traceback: {error_trace}")
        return f"An error occurred while processing your request: {str(e)}"

# Streamlit UI
st.title("🤖 AI Personal Assistant")
st.write("This assistant can help you with:")
st.markdown("- 📅 Creating calendar events")
st.markdown("- 📋 Listing upcoming calendar events")
st.markdown("- 📧 Sending emails")

# Debugging options
debug_mode = st.sidebar.checkbox("Debug Mode")

if debug_mode:
    st.sidebar.subheader("Debug Options")
    test_agent = st.sidebar.radio(
        "Test specific agent directly:", 
        ["None", "calendar_create", "calendar_list", "email"]
    )
    
    credentials_check = st.sidebar.button("Check Credentials")
    if credentials_check:
        st.sidebar.write(f"Google credentials path: {get_google_credentials()}")
        st.sidebar.write(f"Groq client available: {get_groq_client() is not None}")

# User input 
user_input = st.text_area("What can I help you with?", placeholder="e.g., 'Schedule a meeting with John tomorrow at 2 PM' or 'Show my upcoming events'")

# Process Request
if st.button("Process Request"):
    if not user_input:
        st.warning("Please enter a request.")
    else:
        with st.spinner("Processing your request..."):
            if debug_mode and test_agent != "None":
                response = direct_test(user_input, test_agent)
                st.write("Debug: Using direct test mode")
            else:
                response = agent_manager_with_fallback(user_input)
            
            st.markdown(response)

# Main function for local execution
if __name__ == "__main__":
    # This section will only run when directly executing the script
    # The Streamlit UI above will handle web app functionality
    if not st.runtime.exists():
        # Simple command line interface for testing
        query = input("Enter your request: ")
        print("\nProcessing your request...")
        response = agent_manager_with_fallback(query)
        print("\nResponse:", response)