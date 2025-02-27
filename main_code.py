import streamlit as st
import json
import tempfile
from typing import List, Dict, Any, TypedDict
from datetime import datetime, timedelta
import re
import pytz
from langgraph.graph import StateGraph
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

# Move API keys and credentials to environment variables or proper secrets management
def get_groq_client():
    try:
        # Import client only when needed
        from groq import Client
        
        # Better error handling with fallback options
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            st.warning("GROQ API key not found. Some features may not work.")
            return None
            
        return Client(api_key=api_key)
    except ImportError:
        st.error("Groq client library not installed. Install it with: pip install groq")
        return None
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None

def get_google_credentials():
    """Get Google credentials from Streamlit secrets or local file with better error handling."""
    try:
        if "google_credentials" in st.secrets:
            # Create credentials from secrets
            creds_info = {k: v for k, v in st.secrets["google_credentials"].items()}
            
            # Save to temporary file for functions expecting a file path
            temp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            temp.write(json.dumps(creds_info).encode())
            temp.close()
            return temp.name
        elif os.path.exists('token.json'):
            return 'token.json'  # Fallback for local development
        else:
            st.warning("Google credentials not found. Calendar and email features will not work.")
            return None
    except Exception as e:
        st.error(f"Error setting up Google credentials: {str(e)}")
        return None

def safe_json_parse(text, default=None):
    """Safely parse JSON with fallback and improved handling."""
    if not text:
        return default if default is not None else {"agents": ["calendar_list"]}
        
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common JSON formatting issues
        try:
            # Extract just the agents array if possible
            if '"agents":' in text:
                match = re.search(r'"agents":\s*(\[[^\]]+\])', text)
                if match:
                    agents_json = match.group(1)
                    return {"agents": json.loads(agents_json)}
            
            # Try to extract any JSON array
            match = re.search(r'\[(.*?)\]', text)
            if match:
                agents_json = match.group(0)
                return {"agents": json.loads(agents_json)}
        except Exception:
            pass
            
        return default if default is not None else {"agents": ["calendar_list"]}

def create_calendar_event(query: str) -> str:
    """Create a calendar event based on the user query."""
    try:
        credentials_path = get_google_credentials()
        if not credentials_path:
            return "❌ Google credentials not available. Unable to create calendar event."
        
        # Import Google libraries only when needed
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        
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
            # Fallback parser without LLM for basic event creation
            return simple_event_parser(query, timezone)
            
        # Configurable model with fallback options
        model = "gemma2-9b-it"  # Could be configurable
        
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        
        # Parse the LLM response
        try:
            event_details = json.loads(response_content)
        except json.JSONDecodeError as e:
            # Fallback to simple parser if JSON parsing fails
            return simple_event_parser(query, timezone)
        
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

def simple_event_parser(query, timezone):
    """Simple regex-based event parser as fallback when LLM is unavailable"""
    try:
        # Import Google libraries only when needed
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        
        credentials_path = get_google_credentials()
        if not credentials_path:
            return "❌ Google credentials not available. Unable to create calendar event."
            
        calendar = build('calendar', 'v3', 
            credentials=Credentials.from_authorized_user_file(credentials_path, 
                ['https://www.googleapis.com/auth/calendar.events']))
        
        # Extract event title (anything before temporal words)
        title_match = re.match(r'^(.*?)(?:on|tomorrow|next|this|at|from)', query, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = "New Event"  # Default title
            
        # Set date (default to today)
        now = datetime.now(pytz.timezone(timezone))
        event_date = now.date()
        
        # Check for tomorrow
        if "tomorrow" in query.lower():
            event_date = (now + timedelta(days=1)).date()
            
        # Check for specific date
        date_match = re.search(r'on\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)', query, re.IGNORECASE)
        if date_match:
            try:
                day = int(date_match.group(1))
                month_str = date_match.group(2)
                month_dict = {
                    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                    'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
                    'nov': 11, 'november': 11, 'dec': 12, 'december': 12
                }
                month = month_dict.get(month_str.lower(), now.month)
                year = now.year
                
                # Handle next year if month is in the past
                if month < now.month:
                    year += 1
                    
                event_date = datetime(year, month, day).date()
            except (ValueError, KeyError):
                pass
                
        # Extract time
        start_hour, start_minute = 9, 0  # Default to 9 AM
        end_hour, end_minute = 10, 0     # Default to 10 AM
        
        # Check for time specification
        time_match = re.search(r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)', query, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            meridiem = time_match.group(3).lower()
            
            if meridiem == 'pm' and hour < 12:
                hour += 12
                
            start_hour, start_minute = hour, minute
            end_hour, end_minute = hour + 1, minute
            
        # Check for time range
        time_range_match = re.search(r'from\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+to\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)', query, re.IGNORECASE)
        if time_range_match:
            start_hour = int(time_range_match.group(1))
            start_minute = int(time_range_match.group(2)) if time_range_match.group(2) else 0
            start_meridiem = time_range_match.group(3).lower()
            
            end_hour = int(time_range_match.group(4))
            end_minute = int(time_range_match.group(5)) if time_range_match.group(5) else 0
            end_meridiem = time_range_match.group(6).lower()
            
            if start_meridiem == 'pm' and start_hour < 12:
                start_hour += 12
            if end_meridiem == 'pm' and end_hour < 12:
                end_hour += 12
                
        # Extract location
        location = ""
        location_match = re.search(r'at\s+([\w\s]+)(?:$|\.)', query, re.IGNORECASE)
        if location_match and "at" not in location_match.group(1).lower():
            location = location_match.group(1).strip()
            
        # Create datetime objects
        tz = pytz.timezone(timezone)
        start_datetime = tz.localize(datetime.combine(event_date, datetime.min.time().replace(hour=start_hour, minute=start_minute)))
        end_datetime = tz.localize(datetime.combine(event_date, datetime.min.time().replace(hour=end_hour, minute=end_minute)))
        
        # Create event dictionary
        event = {
            'summary': title,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': timezone
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': timezone
            }
        }
        
        if location:
            event['location'] = location
            
        # Insert event into calendar
        result = calendar.events().insert(calendarId='primary', body=event).execute()
        
        # Prepare response
        response_parts = [
            f"✅ Event Scheduled Successfully!",
            f"Event: {title}",
            f"Date: {start_datetime.strftime('%A, %B %d, %Y')}",
            f"Time: {start_datetime.strftime('%I:%M %p')} to {end_datetime.strftime('%I:%M %p')}"
        ]
        
        if location:
            response_parts.append(f"Location: {location}")
            
        response_parts.append(f"View in Calendar: {result.get('htmlLink')}")
        
        return "\n".join(response_parts)
    except Exception as e:
        return f"❌ Failed to create calendar event with fallback parser: {str(e)}"

def list_calendar_events(query: str) -> str:
    """List upcoming calendar events."""
    try:
        credentials_path = get_google_credentials()
        if not credentials_path:
            return "❌ Google credentials not available. Unable to list calendar events."
        
        # Import Google libraries only when needed
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        
        calendar = build('calendar', 'v3', 
            credentials=Credentials.from_authorized_user_file(credentials_path, 
                ['https://www.googleapis.com/auth/calendar.readonly']))
        
        # Extract number of events directly from query with regex
        num_events = 10  # Default
        num_match = re.search(r'(\d+)\s+(?:upcoming\s+)?events', query.lower())
        if num_match:
            try:
                num_events = int(num_match.group(1))
            except ValueError:
                pass
                
        # Handle special cases
        if "my 2 upcoming events" in query.lower() or "list my 2 upcoming events" in query.lower():
            num_events = 2
            
        # If LLM is available, use it for more complex parsing
        if "next few" in query.lower() or "upcoming" in query.lower() and not num_match:
            groq_client = get_groq_client()
            if groq_client:
                prompt = f"""From '{query}' extract number of events to show. Default is 10. Return just the number."""
                
                response = groq_client.chat.completions.create(
                    model="gemma2-9b-it",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response_content = response.choices[0].message.content.strip()
                
                try:
                    extracted_num = int(response_content)
                    if 1 <= extracted_num <= 50:  # Reasonable range check
                        num_events = extracted_num
                except ValueError:
                    pass
        
        # Get date range - default to 30 days
        days_ahead = 30
        if "this week" in query.lower():
            days_ahead = 7
        elif "this month" in query.lower():
            days_ahead = 30
        elif "today" in query.lower():
            days_ahead = 1
            
        events_result = calendar.events().list(
            calendarId='primary',
            timeMin=datetime.utcnow().isoformat() + 'Z',
            timeMax=(datetime.utcnow() + timedelta(days=days_ahead)).isoformat() + 'Z',
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
    




# def send_email(query: str) -> str:
#     """Send an email based on the user query."""
#     try:
#         credentials_path = get_google_credentials()
#         if not credentials_path:
#             return "❌ Google credentials not available. Unable to send email."
        
#         # Import Google libraries only when needed
#         from google.oauth2.credentials import Credentials
#         from googleapiclient.discovery import build
#         from email.mime.text import MIMEText
#         import base64
        
#         # Initialize Gmail service
#         creds = Credentials.from_authorized_user_file(credentials_path, 
#             ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
#         gmail = build('gmail', 'v1', credentials=creds)
        
#         # Extract email address
#         email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', query)
#         if not email_match:
#             return "❌ No email address found in the query! Please include a valid email address."
            
#         recipient = email_match.group()
        
#         # Extract subject if present (anything after "subject:" or "about")
#         subject = "No Subject"
#         subject_match = re.search(r'(?:subject:|about:?)\s+([^\n.]+)', query, re.IGNORECASE)
#         if subject_match:
#             subject = subject_match.group(1).strip()
            
#         # Generate email content
#         groq_client = get_groq_client()
#         if groq_client:
#             # Use LLM for email content
#             prompt = f"""
#             Based on this request: "{query}"
#             Generate a professional email with:
#             1. Professional greeting
#             2. Main message
#             3. Professional closing
#             4. Signature: Best regards,\nMilind Warade

#             Format your response as the email body only.
#             """
            
#             response = groq_client.chat.completions.create(
#                 model="gemma2-9b-it",
#                 messages=[{"role": "user", "content": prompt}]
#             )
            
#             body = response.choices[0].message.content.strip()
#         else:
#             # Fallback to simple email generation
#             body = f"""Hello,

# I'm reaching out regarding your request. {query.replace('send email to ' + recipient, '').replace('about ' + subject, '')}

# Please let me know if you need any further information.

# Best regards,
# Milind Warade"""

#         try:
#             sender = gmail.users().getProfile(userId='me').execute()['emailAddress']
            
#             message = MIMEText(body)
#             message['to'] = recipient
#             message['from'] = sender
#             message['subject'] = subject
            
#             raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#             send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
            
#             final_response = f"""✅ Email sent successfully!
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
#             return final_response
#         except Exception as e:
#             return f"❌ Email sending failed: {e}"
#     except Exception as e:
#         return f"❌ Failed to send email: {str(e)}"

# Gmail Agent Tool
# def generate_template_email(recipient, subject, content_info):
#     """Generate a template-based email when LLM generation fails."""
#     print(f"Debug: Generating template email with content_info: {content_info}")
    
#     # Clean up content info to remove any redundant phrases
#     clean_content = re.sub(r'(?:write|draft|compose|send)\s+(?:an|a)?\s*email\s+(?:about|regarding|on|with)?', '', content_info, flags=re.IGNORECASE)
#     clean_content = clean_content.strip()
    
#     # Create a simple structured email
#     body_parts = [
#         "Hello,",
#         "",
#         f"I'm reaching out regarding {clean_content}.",
#         "",
#         "Please let me know if you need any further information.",
#         "",
#         "Best regards,",
#         "Milind Warade"
#     ]
    
#     return "\n".join(body_parts)

# def send_email(query: str) -> str:
#     """Send an email based on the user query with improved content generation and error handling."""
#     print(f"Debug: send_email called with query: {query}")
#     try:
#         credentials_path = get_google_credentials()
#         print(f"Debug: Using credentials from: {credentials_path}")
        
#         if not credentials_path:
#             print("Debug: No valid credentials found")
#             return "❌ Google credentials not available. Unable to send email."
#         ##Import Google libraries only when needed
#         from google.oauth2.credentials import Credentials
#         from googleapiclient.discovery import build
#         from email.mime.text import MIMEText
#         import base64
        
#         # Initialize Gmail service
#         creds = Credentials.from_authorized_user_file(credentials_path, 
#             ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
#         gmail = build('gmail', 'v1', credentials=creds)
#         print("Debug: Gmail service built successfully")
        
#         # Extract email address with more flexible regex
#         email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
#         email_matches = re.findall(email_pattern, query)
        
#         if not email_matches:
#             print("Debug: No email address found in query")
#             # Try to find keywords like "to" followed by potential recipients
#             to_match = re.search(r'(?:to|send to|email to)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#             if to_match:
#                 recipient_name = to_match.group(1).strip()
#                 return f"❌ Could not find a valid email address for '{recipient_name}'. Please include a complete email address."
#             return "❌ No email address found in the query! Please include a valid email address."
            
#         recipient = email_matches[0]
#         print(f"Debug: Recipient email extracted: {recipient}")
        
#         # Improved subject extraction with better pattern matching
#         subject = "No Subject"
        
#         # First try to find explicit subject indicators
#         subject_patterns = [
#             r'(?:subject|about|regarding|re|titled)[:|\s]\s*"?([^"\.]+)"?',
#             r'(?:email|message|send)\s+(?:with\s+subject|about|regarding)\s+"?([^"\.]+)"?',
#             r'with\s+(?:subject|title)\s+"?([^"\.]+)"?'
#         ]
        
#         for pattern in subject_patterns:
#             subject_match = re.search(pattern, query, re.IGNORECASE)
#             if subject_match:
#                 subject = subject_match.group(1).strip()
#                 if subject.endswith('"'):
#                     subject = subject[:-1]  # Remove trailing quote if present
#                 print(f"Debug: Subject extracted: {subject}")
#                 break
        
#         # If no subject found with explicit indicators, try to extract from the query
#         if subject == "No Subject":
#             # Look for typical subject phrases without explicit markers
#             content_phrases = [
#                 r'asking (?:him|her|them) about (.*?)(?:$|\.|,)',
#                 r'inquiring about (.*?)(?:$|\.|,)',
#                 r'regarding (.*?)(?:$|\.|,)'
#             ]
            
#             for pattern in content_phrases:
#                 content_match = re.search(pattern, query, re.IGNORECASE)
#                 if content_match:
#                     subject = content_match.group(1).strip()
#                     print(f"Debug: Subject extracted from content phrase: {subject}")
#                     break
                    
#         # Extract content/topic information by cleaning the query
#         # Remove recipient and subject portions for better content extraction
#         content_info = query
#         # Remove recipient portion
#         content_info = re.sub(rf'(?:to|send to|email to)\s+{re.escape(recipient)}', '', content_info, flags=re.IGNORECASE)
#         # Remove subject portion if we found one
#         if subject != "No Subject":
#             subject_pattern = rf'(?:subject|about|regarding|re|titled)[:|\s]\s*"?{re.escape(subject)}"?'
#             content_info = re.sub(subject_pattern, '', content_info, flags=re.IGNORECASE)
#         # Clean up common email request phrases
#         content_info = re.sub(r'send\s+(?:an)?\s*email', '', content_info, flags=re.IGNORECASE)
#         content_info = re.sub(r'write\s+(?:an)?\s*email', '', content_info, flags=re.IGNORECASE)
#         content_info = re.sub(r'compose\s+(?:an)?\s*email', '', content_info, flags=re.IGNORECASE)
#         content_info = content_info.strip()
        
#         print(f"Debug: Final content info extracted: {content_info}")
        
#         # Generate email content
#         groq_client = get_groq_client()
#         if groq_client:
#             print("Debug: Using Groq client for email content generation")
#             # Use LLM for email content with more specific instructions
#             prompt = f"""
#             Generate a professional email based on this request: "{query}"
            
#             From the request, I understand:
#             - Recipient: {recipient}
#             - Subject: {subject}
#             - Content relates to: {content_info}
            
#             Create a concise, professional email that:
#             1. Has an appropriate greeting
#             2. Clearly communicates the main message about {subject if subject != "No Subject" else content_info}
#             3. Includes a professional closing
#             4. Ends with: "Best regards,\\nMilind Warade"
            
#             Format your response as the complete email body only, ready to send.
#             """
            
#             try:
#                 response = groq_client.chat.completions.create(
#                     model="gemma2-9b-it",
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=0.7,
#                     max_tokens=500
#                 )
                
#                 body = response.choices[0].message.content.strip()
#                 print(f"Debug: Generated email body length: {len(body)}")
                
#                 # Fallback if LLM returns empty or very short content
#                 if len(body) < 30:
#                     print("Debug: Generated content too short, using template")
#                     body = generate_template_email(recipient, subject, content_info)
#             except Exception as e:
#                 print(f"Debug: Error generating email content with LLM: {str(e)}")
#                 # Fallback to template-based email
#                 body = generate_template_email(recipient, subject, content_info)
#         else:
#             print("Debug: No Groq client available, using template email")
#             # Use template-based email generation when LLM is unavailable
#             body = generate_template_email(recipient, subject, content_info)
        
#         try:
#             # Get sender email
#             profile_response = gmail.users().getProfile(userId='me').execute()
#             sender = profile_response['emailAddress']
#             print(f"Debug: Sender email retrieved: {sender}")
            
#             # Create and send message
#             message = MIMEText(body)
#             message['to'] = recipient
#             message['from'] = sender
#             message['subject'] = subject
            
#             raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#             send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
#             print(f"Debug: Email sent successfully, message ID: {send_result.get('id', 'unknown')}")
            
#             # Format response with email details
#             final_response = f"""✅ Email sent successfully!
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
#             return final_response
#         except Exception as e:
#             error_trace = traceback.format_exc()
#             print(f"Debug: Error sending email: {str(e)}")
#             print(f"Debug: Traceback: {error_trace}")
#             return f"❌ Email sending failed: {str(e)}"
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"Debug: Error in send_email: {str(e)}")
#         print(f"Debug: Traceback: {error_trace}")
#         return f"❌ Failed to send email: {str(e)}"

#1

# def generate_template_email(recipient, recipient_name, subject, content_info):
#     """Generate a template-based email when LLM generation fails."""
#     print(f"Debug: Generating template email with content_info: {content_info}")
    
#     # Clean up content info to remove any redundant phrases and email mentions
#     clean_content = re.sub(r'(?:write|draft|compose|send)\s+(?:an|a)?\s*email\s+(?:about|regarding|on|with)?', '', content_info, flags=re.IGNORECASE)
#     clean_content = re.sub(r'(?:his|her|their)\s+email\s+(?:is|address\s+is)\s+[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', clean_content, flags=re.IGNORECASE)
#     clean_content = re.sub(r'(?:to|ask|tell|email)\s+([A-Z][a-z]+)(?:\s|,|\.)', '', clean_content, flags=re.IGNORECASE)
#     clean_content = clean_content.strip()
    
#     # Extract project or topic if available
#     project_match = re.search(r'(?:project|task|assignment)\s+([a-zA-Z0-9\s]+)', clean_content, re.IGNORECASE)
#     project_name = project_match.group(1) if project_match else "the project"
    
#     # Determine if this is an update request
#     is_update_request = "update" in clean_content.lower() or "progress" in clean_content.lower() or "status" in clean_content.lower()
    
#     # Create a proper professional email
#     body_parts = [
#         f"Hello {recipient_name},",
#         "",
#     ]
    
#     if is_update_request:
#         body_parts.extend([
#             f"I hope this email finds you well. I'm writing to request an update on Project {project_name}.",
#             "",
#             "Could you please provide the current status and any recent progress on this project? I would appreciate details on:",
#             "- Work completed so far",
#             "- Any challenges or blockers you're facing",
#             "- Timeline for upcoming milestones",
#             ""
#         ])
#     else:
#         body_parts.extend([
#             f"I hope this email finds you well. I'm reaching out regarding Project {project_name}.",
#             "",
#             f"{clean_content}",
#             ""
#         ])
    
#     body_parts.extend([
#         "I would appreciate your response at your earliest convenience. If you need any additional information or have questions, please don't hesitate to ask.",
#         "",
#         "Best regards,",
#         "Milind Warade"
#     ])
    
#     return "\n".join(body_parts)

# def clean_query_for_content(query):
#     """Clean the query to extract only relevant content."""
#     # Remove email addresses
#     cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', query)
#     # Remove email related commands
#     cleaned = re.sub(r'(?:write|draft|compose|send)\s+(?:an|a)?\s*email\s+(?:about|regarding|on|with)?', '', cleaned, flags=re.IGNORECASE)
#     # Remove "to [Name]" patterns
#     cleaned = re.sub(r'to\s+([A-Z][a-z]+)(?:\s|,|\.)', '', cleaned, flags=re.IGNORECASE)
#     # Remove email address references
#     cleaned = re.sub(r'(?:his|her|their)\s+email\s+(?:is|address\s+is)', '', cleaned, flags=re.IGNORECASE)
#     # Remove asking/telling mentions
#     cleaned = re.sub(r'(?:asking|telling)\s+(?:him|her|them)', '', cleaned, flags=re.IGNORECASE)
#     return cleaned.strip()

# def extract_recipient_name(query, recipient_email):
#     """Extract the recipient's name from the query."""
#     # Look for patterns like "to John" or "ask Sarah"
#     name_pattern = r'(?:to|ask|tell|email)\s+([A-Z][a-z]+)(?:\s|,|\.)'
#     name_match = re.search(name_pattern, query, re.IGNORECASE)
    
#     if name_match:
#         return name_match.group(1)
    
#     # Try to extract name from email (before @ symbol)
#     email_name = recipient_email.split('@')[0]
#     # Convert email format to proper name (e.g., john.doe -> John)
#     if '.' in email_name:
#         parts = email_name.split('.')
#         return ' '.join(part.capitalize() for part in parts)
#     return email_name.capitalize()

# def extract_subject_from_query(query):
#     """Extract a meaningful subject from the query."""
#     # First check for explicit subject mentions
#     subject_patterns = [
#         r'(?:subject|about|regarding|re|titled)[:|\s]\s*"?([^"\.]+)"?',
#         r'(?:email|message|send)\s+(?:with\s+subject|about|regarding)\s+"?([^"\.]+)"?',
#         r'with\s+(?:subject|title)\s+"?([^"\.]+)"?'
#     ]
    
#     for pattern in subject_patterns:
#         subject_match = re.search(pattern, query, re.IGNORECASE)
#         if subject_match:
#             return subject_match.group(1).strip()
    
#     # Look for project mentions
#     project_match = re.search(r'(?:project|task)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     if project_match:
#         project = project_match.group(1).strip()
#         # Check if this is an update request
#         if re.search(r'update|progress|status', query, re.IGNORECASE):
#             return f"Update on Project {project}"
#         return f"Regarding Project {project}"
    
#     # Check for update requests
#     if re.search(r'update|progress|status', query, re.IGNORECASE):
#         return "Project Update Request"
    
#     # Generic fallback
#     return "Project Information Request"

# def send_email(query: str) -> str:
#     """Send an email based on the user query with improved content generation and error handling."""
#     print(f"Debug: send_email called with query: {query}")
#     try:
#         credentials_path = get_google_credentials()
#         print(f"Debug: Using credentials from: {credentials_path}")
        
#         if not credentials_path:
#             print("Debug: No valid credentials found")
#             return "❌ Google credentials not available. Unable to send email."
        
#         ##Import Google libraries only when needed
#         from google.oauth2.credentials import Credentials
#         from googleapiclient.discovery import build
#         from email.mime.text import MIMEText
#         import base64
        
#         # Initialize Gmail service
#         creds = Credentials.from_authorized_user_file(credentials_path, 
#             ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
#         gmail = build('gmail', 'v1', credentials=creds)
#         print("Debug: Gmail service built successfully")
        
#         # Extract email address with more flexible regex
#         email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
#         email_matches = re.findall(email_pattern, query)
        
#         if not email_matches:
#             print("Debug: No email address found in query")
#             # Try to find keywords like "to" followed by potential recipients
#             to_match = re.search(r'(?:to|send to|email to)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#             if to_match:
#                 recipient_name = to_match.group(1).strip()
#                 return f"❌ Could not find a valid email address for '{recipient_name}'. Please include a complete email address."
#             return "❌ No email address found in the query! Please include a valid email address."
            
#         recipient = email_matches[0]
#         print(f"Debug: Recipient email extracted: {recipient}")
        
#         # Extract recipient name
#         recipient_name = extract_recipient_name(query, recipient)
#         print(f"Debug: Recipient name extracted: {recipient_name}")
        
#         # Extract subject from query
#         subject = extract_subject_from_query(query)
#         print(f"Debug: Subject extracted: {subject}")
                    
#         # Clean the query to extract content information
#         content_info = clean_query_for_content(query)
#         print(f"Debug: Cleaned content info: {content_info}")
        
#         # Generate email content
#         groq_client = get_groq_client()
#         if groq_client:
#             print("Debug: Using Groq client for email content generation")
#             # Use LLM for email content with more specific instructions
#             prompt = f"""
#             You are a professional assistant composing an email.
            
#             Create a professional email with the following information:
#             - Recipient name: {recipient_name}
#             - Subject: {subject}
#             - Purpose: {content_info}
            
#             Guidelines:
#             1. Write a clear, concise, and professional email
#             2. Start with "Hello {recipient_name},"
#             3. DO NOT include any placeholder text
#             4. DO NOT repeat the email address in the body
#             5. DO NOT mention anything about "composing an email" in the content
#             6. DO NOT format as "I'm reaching out regarding compose and to John asking him..."
#             7. If this is an update request, ask for specific updates on the project status
#             8. End with "I would appreciate your response at your earliest convenience."
#             9. Use the signature "Best regards,\\nMilind Warade"
            
#             Your email should be ready to send as-is with no modifications needed.
#             """
            
#             try:
#                 response = groq_client.chat.completions.create(
#                     model="gemma2-9b-it",
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=0.7,
#                     max_tokens=500
#                 )
                
#                 body = response.choices[0].message.content.strip()
#                 print(f"Debug: Generated email body length: {len(body)}")
                
#                 # Verify email quality - check for template phrases and raw query content
#                 poor_quality_indicators = [
#                     "compose and",
#                     "asking him",
#                     "send an email",
#                     "[Your specific message here]",
#                     "[Insert details]"
#                 ]
                
#                 is_poor_quality = len(body) < 100 or any(indicator in body for indicator in poor_quality_indicators)
                
#                 if is_poor_quality:
#                     print("Debug: Generated content is poor quality, using template instead")
#                     body = generate_template_email(recipient, recipient_name, subject, content_info)
#             except Exception as e:
#                 print(f"Debug: Error generating email content with LLM: {str(e)}")
#                 # Fallback to template-based email
#                 body = generate_template_email(recipient, recipient_name, subject, content_info)
#         else:
#             print("Debug: No Groq client available, using template email")
#             # Use template-based email generation when LLM is unavailable
#             body = generate_template_email(recipient, recipient_name, subject, content_info)
        
#         try:
#             # Get sender email
#             profile_response = gmail.users().getProfile(userId='me').execute()
#             sender = profile_response['emailAddress']
#             print(f"Debug: Sender email retrieved: {sender}")
            
#             # Create and send message
#             message = MIMEText(body)
#             message['to'] = recipient
#             message['from'] = sender
#             message['subject'] = subject
            
#             raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#             send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
#             print(f"Debug: Email sent successfully, message ID: {send_result.get('id', 'unknown')}")
            
#             # Format response with email details
#             final_response = f"""✅ Email sent successfully!
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
#             return final_response
#         except Exception as e:
#             error_trace = traceback.format_exc()
#             print(f"Debug: Error sending email: {str(e)}")
#             print(f"Debug: Traceback: {error_trace}")
#             return f"❌ Email sending failed: {str(e)}"
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"Debug: Error in send_email: {str(e)}")
#         print(f"Debug: Traceback: {error_trace}")
#         return f"❌ Failed to send email: {str(e)}"

#2

# def generate_template_email(recipient, recipient_name, subject, content_info):
#     """Generate a template-based email when LLM generation fails."""
#     print(f"Debug: Generating template email with content_info: {content_info}")
    
#     # Clean up content info to remove any redundant phrases and email mentions
#     clean_content = re.sub(r'(?:write|draft|compose|send)\s+(?:an|a)?\s*email\s+(?:about|regarding|on|with)?', '', content_info, flags=re.IGNORECASE)
#     clean_content = re.sub(r'(?:his|her|their)\s+email\s+(?:is|address\s+is)\s+[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', clean_content, flags=re.IGNORECASE)
#     clean_content = re.sub(r'(?:to|ask|tell|email)\s+([A-Z][a-z]+)(?:\s|,|\.)', '', clean_content, flags=re.IGNORECASE)
    
#     # Extract key topics like "benefits of exercise"
#     topic_match = re.search(r'benefits\s+of\s+([a-zA-Z0-9\s]+)', clean_content, re.IGNORECASE)
#     if topic_match:
#         topic = f"benefits of {topic_match.group(1)}"
#     else:
#         # Extract any meaningful phrases
#         topic_match = re.search(r'(about|regarding|on|telling)\s+([a-zA-Z0-9\s]+)', clean_content, re.IGNORECASE)
#         topic = topic_match.group(2) if topic_match else "the topic"
    
#     clean_content = clean_content.strip()
    
#     # Extract project or topic if available
#     project_match = re.search(r'(?:project|task|assignment)\s+([a-zA-Z0-9\s]+)', clean_content, re.IGNORECASE)
#     project_name = project_match.group(1) if project_match else None
    
#     # Determine if this is an update request, benefits discussion, or general inquiry
#     is_update_request = "update" in clean_content.lower() or "progress" in clean_content.lower() or "status" in clean_content.lower()
#     is_benefits_discussion = "benefits" in clean_content.lower() or "advantages" in clean_content.lower()
    
#     # Create a proper professional email
#     body_parts = [
#         f"Hello {recipient_name},",
#         "",
#     ]
    
#     if is_update_request and project_name:
#         body_parts.extend([
#             f"I hope this email finds you well. I'm writing to request an update on Project {project_name}.",
#             "",
#             "Could you please provide the current status and any recent progress on this project? I would appreciate details on:",
#             "- Work completed so far",
#             "- Any challenges or blockers you're facing",
#             "- Timeline for upcoming milestones",
#             ""
#         ])
#     elif is_benefits_discussion:
#         body_parts.extend([
#             "I hope this email finds you well. I'm reaching out regarding the benefits of exercise.",
#             "",
#             "Regular physical activity offers numerous advantages for both physical and mental health, including:",
#             "- Improved cardiovascular health and reduced risk of heart disease",
#             "- Increased energy levels and better sleep quality",
#             "- Reduced stress and anxiety",
#             "- Enhanced mood and cognitive function",
#             "- Maintenance of healthy weight",
#             "",
#             "Even small amounts of daily activity can lead to significant long-term health benefits."
#         ])
#     else:
#         # For general inquiries or when specific patterns aren't detected
#         if project_name:
#             body_parts.extend([
#                 f"I hope this email finds you well. I'm reaching out regarding Project {project_name}.",
#                 ""
#             ])
#         else:
#             body_parts.extend([
#                 f"I hope this email finds you well. I'm reaching out regarding {topic}.",
#                 ""
#             ])
        
#         # Add the cleaned content if it's substantial
#         if len(clean_content) > 10:
#             body_parts.extend([clean_content, ""])
    
#     body_parts.extend([
#         "I would appreciate your response at your earliest convenience. If you need any additional information or have questions, please don't hesitate to ask.",
#         "",
#         "Best regards,",
#         "Milind Warade"
#     ])
    
#     return "\n".join(body_parts)

# def clean_query_for_content(query):
#     """Clean the query to extract only relevant content."""
#     # Remove email addresses
#     cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', query)
    
#     # Remove email related commands but be careful not to remove content
#     cleaned = re.sub(r'(?:write|draft|compose|send)\s+(?:an|a)?\s*email\s+(?:about|regarding|on|with)?', '', cleaned, flags=re.IGNORECASE)
    
#     # Extract the main content more carefully
#     # If "telling the benefits of" pattern exists, preserve it
#     if re.search(r'telling\s+the\s+benefits\s+of', cleaned, flags=re.IGNORECASE):
#         main_content = re.search(r'telling\s+the\s+benefits\s+of\s+(.+?)(?:\.|$)', cleaned, flags=re.IGNORECASE)
#         if main_content:
#             return f"the benefits of {main_content.group(1)}".strip()
    
#     # Remove "to [Name]" patterns
#     cleaned = re.sub(r'to\s+([A-Z][a-z]+)(?:\s|,|\.)', '', cleaned, flags=re.IGNORECASE)
    
#     # Remove email address references
#     cleaned = re.sub(r'(?:his|her|their)\s+email\s+(?:is|address\s+is)', '', cleaned, flags=re.IGNORECASE)
    
#     # Remove asking/telling mentions but preserve what comes after
#     telling_match = re.search(r'(?:asking|telling)\s+(?:him|her|them)\s+(.+)', cleaned, flags=re.IGNORECASE)
#     if telling_match:
#         return telling_match.group(1).strip()
    
#     return cleaned.strip()

# def extract_subject_from_query(query):
#     """Extract a meaningful subject from the query."""
#     # First check for explicit subject mentions
#     subject_patterns = [
#         r'(?:subject|about|regarding|re|titled)[:|\s]\s*"?([^"\.]+)"?',
#         r'(?:email|message|send)\s+(?:with\s+subject|about|regarding)\s+"?([^"\.]+)"?',
#         r'with\s+(?:subject|title)\s+"?([^"\.]+)"?'
#     ]
    
#     for pattern in subject_patterns:
#         subject_match = re.search(pattern, query, re.IGNORECASE)
#         if subject_match:
#             return subject_match.group(1).strip()
    
#     # Check for benefits of something
#     benefits_match = re.search(r'benefits\s+of\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     if benefits_match:
#         return f"Benefits of {benefits_match.group(1).capitalize()}"
    
#     # Look for project mentions
#     project_match = re.search(r'(?:project|task)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     if project_match:
#         project = project_match.group(1).strip()
#         # Check if this is an update request
#         if re.search(r'update|progress|status', query, re.IGNORECASE):
#             return f"Update on Project {project}"
#         return f"Regarding Project {project}"
    
#     # Check for "telling" pattern
#     telling_match = re.search(r'telling\s+(?:him|her|them)\s+(?:about|regarding)?\s*(.+?)(?:\.|$)', query, re.IGNORECASE)
#     if telling_match:
#         topic = telling_match.group(1).strip()
#         return f"Information about {topic}"
    
#     # Check for update requests
#     if re.search(r'update|progress|status', query, re.IGNORECASE):
#         return "Project Update Request"
    
#     # Generic fallback
#     return "Information Request"

# def extract_recipient_name(query, recipient_email):
#     """Extract the recipient's name from the query or email address."""
#     # Look for patterns like "to John" or "ask Sarah"
#     name_pattern = r'(?:to|ask|tell|email)\s+([A-Z][a-z]+)(?:\s|,|\.)'
#     name_match = re.search(name_pattern, query, re.IGNORECASE)
    
#     if name_match:
#         return name_match.group(1)
    
#     # Try to extract name from email (before @ symbol)
#     email_name = recipient_email.split('@')[0]
#     # Convert email format to proper name (e.g., john.doe -> John Doe)
#     if '.' in email_name:
#         parts = email_name.split('.')
#         return ' '.join(part.capitalize() for part in parts)
#     return email_name.capitalize()

# def send_email(query: str) -> str:
#     """Send an email based on the user query with improved content generation and error handling."""
#     print(f"Debug: send_email called with query: {query}")
#     try:
#         credentials_path = get_google_credentials()
#         print(f"Debug: Using credentials from: {credentials_path}")
        
#         if not credentials_path:
#             print("Debug: No valid credentials found")
#             return "❌ Google credentials not available. Unable to send email."
        
#         ##Import Google libraries only when needed
#         from google.oauth2.credentials import Credentials
#         from googleapiclient.discovery import build
#         from email.mime.text import MIMEText
#         import base64
        
#         # Initialize Gmail service
#         creds = Credentials.from_authorized_user_file(credentials_path, 
#             ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
#         gmail = build('gmail', 'v1', credentials=creds)
#         print("Debug: Gmail service built successfully")
        
#         # Extract email address with more flexible regex
#         email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
#         email_matches = re.findall(email_pattern, query)
        
#         if not email_matches:
#             print("Debug: No email address found in query")
#             # Try to find keywords like "to" followed by potential recipients
#             to_match = re.search(r'(?:to|send to|email to)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#             if to_match:
#                 recipient_name = to_match.group(1).strip()
#                 return f"❌ Could not find a valid email address for '{recipient_name}'. Please include a complete email address."
#             return "❌ No email address found in the query! Please include a valid email address."
            
#         recipient = email_matches[0]
#         print(f"Debug: Recipient email extracted: {recipient}")
        
#         # Extract recipient name
#         recipient_name = extract_recipient_name(query, recipient)
#         print(f"Debug: Recipient name extracted: {recipient_name}")
        
#         # Extract subject from query
#         subject = extract_subject_from_query(query)
#         print(f"Debug: Subject extracted: {subject}")
                    
#         # Clean the query to extract content information
#         content_info = clean_query_for_content(query)
#         print(f"Debug: Cleaned content info: {content_info}")
        
#         # Generate email content
#         groq_client = get_groq_client()
#         if groq_client:
#             print("Debug: Using Groq client for email content generation")
#             # Use LLM for email content with more specific instructions
#             prompt = f"""
#             You are a professional assistant composing an email.
            
#             Create a professional email with the following information:
#             - Recipient name: {recipient_name}
#             - Subject: {subject}
#             - Purpose: {content_info}
            
#             Guidelines:
#             1. Write a clear, concise, and professional email
#             2. Start with "Hello {recipient_name},"
#             3. DO NOT include any placeholder text
#             4. DO NOT repeat the email address in the body
#             5. DO NOT mention anything about "composing an email" in the content
#             6. DO NOT format as "I'm reaching out regarding compose and to John asking him..."
#             7. If this is an update request, ask for specific updates on the project status
#             8. If this is about benefits of exercise, include some key health benefits
#             9. End with "I would appreciate your response at your earliest convenience."
#             10. Use the signature "Best regards,\\nMilind Warade"
            
#             Your email should be ready to send as-is with no modifications needed.
#             """
            
#             try:
#                 response = groq_client.chat.completions.create(
#                     model="gemma2-9b-it",
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=0.7,
#                     max_tokens=500
#                 )
                
#                 body = response.choices[0].message.content.strip()
#                 print(f"Debug: Generated email body length: {len(body)}")
                
#                 # Verify email quality - check for template phrases and raw query content
#                 poor_quality_indicators = [
#                     "compose and",
#                     "asking him",
#                     "send an email",
#                     "[Your specific message here]",
#                     "[Insert details]"
#                 ]
                
#                 is_poor_quality = len(body) < 100 or any(indicator in body for indicator in poor_quality_indicators)
                
#                 if is_poor_quality:
#                     print("Debug: Generated content is poor quality, using template instead")
#                     body = generate_template_email(recipient, recipient_name, subject, content_info)
#             except Exception as e:
#                 print(f"Debug: Error generating email content with LLM: {str(e)}")
#                 # Fallback to template-based email
#                 body = generate_template_email(recipient, recipient_name, subject, content_info)
#         else:
#             print("Debug: No Groq client available, using template email")
#             # Use template-based email generation when LLM is unavailable
#             body = generate_template_email(recipient, recipient_name, subject, content_info)
        
#         try:
#             # Get sender email
#             profile_response = gmail.users().getProfile(userId='me').execute()
#             sender = profile_response['emailAddress']
#             print(f"Debug: Sender email retrieved: {sender}")
            
#             # Create and send message
#             message = MIMEText(body)
#             message['to'] = recipient
#             message['from'] = sender
#             message['subject'] = subject
            
#             raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#             send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
#             print(f"Debug: Email sent successfully, message ID: {send_result.get('id', 'unknown')}")
            
#             # Format response with email details
#             final_response = f"""✅ Email sent successfully!
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
#             return final_response
#         except Exception as e:
#             error_trace = traceback.format_exc()
#             print(f"Debug: Error sending email: {str(e)}")
#             print(f"Debug: Traceback: {error_trace}")
#             return f"❌ Email sending failed: {str(e)}"
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"Debug: Error in send_email: {str(e)}")
#         print(f"Debug: Traceback: {error_trace}")
#         return f"❌ Failed to send email: {str(e)}"

#3

# import re
# import traceback
# from email.mime.text import MIMEText
# import base64

# def get_google_credentials():
#     """Placeholder function - implement to return path to credentials file"""
#     # Replace with actual implementation
#     return "path/to/credentials.json"

# def get_groq_client():
#     """Placeholder function - implement to return Groq client"""
#     # Replace with actual implementation
#     return None

# def generate_email_content(recipient_name, subject, query):
#     """Generate professional email content for office environment queries."""
#     # Extract the main content/purpose from the query
#     content = extract_main_content(query)
    
#     # Standard professional email template
#     body = [
#         f"Hello {recipient_name},",
#         "",
#         "I hope this email finds you well."
#     ]
    
#     # Add appropriate content based on query type
#     if "access" in query.lower() and "software" in query.lower():
#         body.extend([
#             "",
#             "I am writing to request access to the new software. I would need this access to perform my duties effectively.",
#             "",
#             "Could you please provide me with the necessary credentials or instructions to access the system?"
#         ])
#     elif "update" in query.lower() or "status" in query.lower() or "progress" in query.lower():
#         # For project update requests
#         project = extract_project_name(query) or "the project"
#         body.extend([
#             "",
#             f"I am writing to request an update on {project}.",
#             "",
#             "Could you please share with me:",
#             "- Current progress and status",
#             "- Any challenges or blockers faced",
#             "- Expected timeline for completion"
#         ])
#     elif "meeting" in query.lower() or "schedule" in query.lower():
#         # For meeting requests
#         body.extend([
#             "",
#             "I would like to schedule a meeting to discuss our current projects and priorities.",
#             "",
#             "Please let me know your availability for next week so we can find a suitable time."
#         ])
#     elif "document" in query.lower() or "file" in query.lower():
#         # For document requests
#         body.extend([
#             "",
#             "I am writing to request the documents related to our recent project.",
#             "",
#             "Could you please share these files with me at your earliest convenience?"
#         ])
#     elif "benefits" in query.lower():
#         topic = extract_benefit_topic(query) or "the program"
#         body.extend([
#             "",
#             f"I wanted to share with you some key benefits of {topic}:",
#             "",
#             "- Improved efficiency and productivity",
#             "- Cost savings and resource optimization",
#             "- Enhanced collaboration and communication",
#             "- Better outcomes and quality results",
#             "",
#             f"I believe implementing {topic} would be valuable for our team."
#         ])
#     else:
#         # Generic professional content for any other query
#         body.extend([
#             "",
#             f"{content}",
#             "",
#             "I look forward to your insights on this matter."
#         ])
    
#     body.extend([
#         "",
#         "I would appreciate your response at your earliest convenience. If you need any additional information or have questions, please don't hesitate to ask.",
#         "",
#         "Best regards,",
#         "Milind Warade"
#     ])
    
#     return "\n".join(body)

# def extract_main_content(query):
#     """Extract the main content/purpose from the query."""
#     # Remove email addresses
#     cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', query)
    
#     # Remove email related commands
#     cleaned = re.sub(r'(?:send|write|compose|email)\s+(?:an|a)?\s*email\s+(?:to|about|regarding|on|with)?', '', cleaned, flags=re.IGNORECASE)
    
#     # Remove "requesting" if it exists
#     cleaned = re.sub(r'requesting\s+', '', cleaned, flags=re.IGNORECASE)
    
#     # This is our main content
#     return cleaned.strip()

# def extract_project_name(query):
#     """Extract project name from query if available."""
#     match = re.search(r'(?:project|task)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     return match.group(1).strip() if match else None

# def extract_benefit_topic(query):
#     """Extract topic for benefits discussion."""
#     match = re.search(r'benefits\s+of\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     return match.group(1).strip() if match else None

# def extract_recipient_name(email):
#     """Extract recipient name from email address."""
#     name = email.split('@')[0]
#     # Convert email format to proper name (e.g., john.doe -> John Doe)
#     if '.' in name:
#         parts = name.split('.')
#         return ' '.join(part.capitalize() for part in parts)
#     return name.capitalize()

# def generate_subject(query):
#     """Generate appropriate subject line based on query content."""
#     if "access" in query.lower() and "software" in query.lower():
#         return "Request for Software Access"
#     elif "update" in query.lower() or "status" in query.lower() or "progress" in query.lower():
#         project = extract_project_name(query)
#         return f"Status Update Request: {project}" if project else "Project Status Update Request"
#     elif "meeting" in query.lower() or "schedule" in query.lower():
#         return "Meeting Request"
#     elif "document" in query.lower() or "file" in query.lower():
#         return "Document Request"
#     elif "benefits" in query.lower():
#         topic = extract_benefit_topic(query)
#         return f"Benefits of {topic.capitalize()}" if topic else "Information on Benefits"
#     else:
#         # Generic subject that uses the first few words of content
#         content = extract_main_content(query)
#         words = content.split()[:3]  # First 3 words
#         if words:
#             return " ".join(words).capitalize()
#         return "Information Request"

# def send_email(query: str) -> str:
#     """Send an email based on the user query."""
#     print(f"Debug: send_email called with query: {query}")
#     try:
#         # Extract email address
#         email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
#         email_matches = re.findall(email_pattern, query)
        
#         if not email_matches:
#             return "❌ No email address found in the query! Please include a valid email address."
            
#         recipient = email_matches[0]
#         print(f"Debug: Recipient email extracted: {recipient}")
        
#         # Extract recipient name from email
#         recipient_name = extract_recipient_name(recipient)
#         print(f"Debug: Recipient name extracted: {recipient_name}")
        
#         # Generate a relevant subject line
#         subject = generate_subject(query)
#         print(f"Debug: Subject generated: {subject}")
        
#         # Generate email content
#         body = generate_email_content(recipient_name, subject, query)
#         print(f"Debug: Email body generated with length: {len(body)}")
        
#         try:
#             # Get Google credentials and initialize Gmail API
#             credentials_path = get_google_credentials()
            
#             if not credentials_path:
#                 print("Debug: No valid credentials found")
#                 # Since we can't send the email, just return as if it was sent for demo purposes
#                 return f"""✅ Email would be sent (demo mode):
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
            
#             # Import Google libraries only when needed
#             from google.oauth2.credentials import Credentials
#             from googleapiclient.discovery import build
            
#             # Initialize Gmail service
#             creds = Credentials.from_authorized_user_file(credentials_path, 
#                 ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
#             gmail = build('gmail', 'v1', credentials=creds)
            
#             # Get sender email
#             profile_response = gmail.users().getProfile(userId='me').execute()
#             sender = profile_response['emailAddress']
            
#             # Create and send message
#             message = MIMEText(body)
#             message['to'] = recipient
#             message['from'] = sender
#             message['subject'] = subject
            
#             raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#             send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
            
#             # Format success response
#             return f"""✅ Email sent successfully!
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
            
#         except Exception as e:
#             error_trace = traceback.format_exc()
#             print(f"Debug: Error sending email: {str(e)}")
#             print(f"Debug: Traceback: {error_trace}")
            
#             # For demonstration, return as if email was sent
#             return f"""✅ Email would be sent (demo mode):
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
            
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"Debug: Error in send_email: {str(e)}")
#         print(f"Debug: Traceback: {error_trace}")
#         return f"❌ Failed to process email request: {str(e)}"


#4

# import re
# import traceback
# from email.mime.text import MIMEText
# import base64

# def get_google_credentials():
#     """Placeholder function - implement to return path to credentials file"""
#     # Replace with actual implementation
#     return "path/to/credentials.json"

# def get_groq_client():
#     """Placeholder function - implement to return Groq client"""
#     # Replace with actual implementation
#     return None

# def generate_email_content(recipient_name, subject, query):
#     """Generate professional email content for office environment queries."""
#     # Extract the main content/purpose from the query
#     content = extract_main_content(query)
    
#     # Standard professional email template
#     body = [
#         f"Hello {recipient_name},",
#         "",
#         "I hope this email finds you well."
#     ]
    
#     # Add appropriate content based on query type
#     if "access" in query.lower() and "software" in query.lower():
#         body.extend([
#             "",
#             "I am writing to request access to the new software. I would need this access to perform my duties effectively.",
#             "",
#             "Could you please provide me with the necessary credentials or instructions to access the system?"
#         ])
#     elif "update" in query.lower() or "status" in query.lower() or "progress" in query.lower():
#         # For project update requests
#         project = extract_project_name(query) or "the project"
#         body.extend([
#             "",
#             f"I am writing to request an update on {project}.",
#             "",
#             "Could you please share with me:",
#             "- Current progress and status",
#             "- Any challenges or blockers faced",
#             "- Expected timeline for completion"
#         ])
#     elif "meeting" in query.lower() or "schedule" in query.lower():
#         # For meeting requests
#         body.extend([
#             "",
#             "I would like to schedule a meeting to discuss our current projects and priorities.",
#             "",
#             "Please let me know your availability for next week so we can find a suitable time."
#         ])
#     elif "document" in query.lower() or "file" in query.lower():
#         # For document requests
#         body.extend([
#             "",
#             "I am writing to request the documents related to our recent project.",
#             "",
#             "Could you please share these files with me at your earliest convenience?"
#         ])
#     elif "benefits" in query.lower():
#         topic = extract_benefit_topic(query) or "the program"
#         body.extend([
#             "",
#             f"I wanted to share with you some key benefits of {topic}:",
#             "",
#             "- Improved efficiency and productivity",
#             "- Cost savings and resource optimization",
#             "- Enhanced collaboration and communication",
#             "- Better outcomes and quality results",
#             "",
#             f"I believe implementing {topic} would be valuable for our team."
#         ])
#     else:
#         # Generic professional content for any other query
#         body.extend([
#             "",
#             f"{content}",
#             "",
#             "I look forward to your insights on this matter."
#         ])
    
#     body.extend([
#         "",
#         "I would appreciate your response at your earliest convenience. If you need any additional information or have questions, please don't hesitate to ask.",
#         "",
#         "Best regards,",
#         "Milind Warade"
#     ])
    
#     return "\n".join(body)

# def extract_main_content(query):
#     """Extract the main content/purpose from the query."""
#     # Remove email addresses
#     cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', query)
    
#     # Remove email related commands
#     cleaned = re.sub(r'(?:send|write|compose|email)\s+(?:an|a)?\s*email\s+(?:to|about|regarding|on|with)?', '', cleaned, flags=re.IGNORECASE)
    
#     # Remove "requesting" if it exists
#     cleaned = re.sub(r'requesting\s+', '', cleaned, flags=re.IGNORECASE)
    
#     # This is our main content
#     return cleaned.strip()

# def extract_project_name(query):
#     """Extract project name from query if available."""
#     match = re.search(r'(?:project|task)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     return match.group(1).strip() if match else None

# def extract_benefit_topic(query):
#     """Extract topic for benefits discussion."""
#     match = re.search(r'benefits\s+of\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     return match.group(1).strip() if match else None

# def extract_recipient_name(email):
#     """Extract recipient name from email address."""
#     name = email.split('@')[0]
#     # Convert email format to proper name (e.g., john.doe -> John Doe)
#     if '.' in name:
#         parts = name.split('.')
#         return ' '.join(part.capitalize() for part in parts)
#     return name.capitalize()

# def generate_subject(query):
#     """Generate appropriate subject line based on query content."""
#     if "access" in query.lower() and "software" in query.lower():
#         return "Request for Software Access"
#     elif "update" in query.lower() or "status" in query.lower() or "progress" in query.lower():
#         project = extract_project_name(query)
#         return f"Status Update Request: {project}" if project else "Project Status Update Request"
#     elif "meeting" in query.lower() or "schedule" in query.lower():
#         return "Meeting Request"
#     elif "document" in query.lower() or "file" in query.lower():
#         return "Document Request"
#     elif "benefits" in query.lower():
#         topic = extract_benefit_topic(query)
#         return f"Benefits of {topic.capitalize()}" if topic else "Information on Benefits"
#     else:
#         # Generic subject that uses the first few words of content
#         content = extract_main_content(query)
#         words = content.split()[:3]  # First 3 words
#         if words:
#             return " ".join(words).capitalize()
#         return "Information Request"



# def send_email(query: str) -> str:
#     """Send an email based on the user query."""
#     print(f"Debug: send_email called with query: {query}")
#     try:
#         # Extract email address
#         email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
#         email_matches = re.findall(email_pattern, query)
        
#         if not email_matches:
#             return "❌ No email address found in the query! Please include a valid email address."
            
#         recipient = email_matches[0]
#         print(f"Debug: Recipient email extracted: {recipient}")
        
#         # Get Google credentials
#         credentials_path = get_google_credentials()
#         if not credentials_path:
#             return "❌ Google credentials not available. Unable to send email."
            
#         # Import Google libraries only when needed
#         from google.oauth2.credentials import Credentials
#         from googleapiclient.discovery import build
        
#         # Initialize Gmail service
#         creds = Credentials.from_authorized_user_file(credentials_path, 
#             ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
#         gmail = build('gmail', 'v1', credentials=creds)
        
#         # Extract recipient name from email
#         recipient_name = extract_recipient_name(recipient)
#         print(f"Debug: Recipient name extracted: {recipient_name}")
        
#         # Generate a relevant subject line
#         subject = generate_subject(query)
#         print(f"Debug: Subject generated: {subject}")
        
#         # Generate email content
#         body = generate_email_content(recipient_name, subject, query)
#         print(f"Debug: Email body generated with length: {len(body)}")
        
#         try:
#             # Get sender email
#             profile_response = gmail.users().getProfile(userId='me').execute()
#             sender = profile_response['emailAddress']
            
#             # Create and send message
#             message = MIMEText(body)
#             message['to'] = recipient
#             message['from'] = sender
#             message['subject'] = subject
            
#             raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#             send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
            
#             # Format success response
#             return f"""✅ Email sent successfully!
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
            
#         except Exception as e:
#             error_trace = traceback.format_exc()
#             print(f"Debug: Error sending email: {str(e)}")
#             print(f"Debug: Traceback: {error_trace}")
#             return f"❌ Email sending failed: {str(e)}"
            
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"Debug: Error in send_email: {str(e)}")
#         print(f"Debug: Traceback: {error_trace}")
#         return f"❌ Failed to process email request: {str(e)}"

#5

# def extract_main_content(query):
#     """Extract the main content/purpose from the query."""
#     # Remove email addresses
#     cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', query)
    
#     # Remove email related commands
#     cleaned = re.sub(r'(?:send|write|compose|email)\s+(?:an|a)?\s*email\s+(?:to|about|regarding|on|with)?', '', cleaned, flags=re.IGNORECASE)
    
#     # Remove "requesting" if it exists
#     cleaned = re.sub(r'requesting\s+', '', cleaned, flags=re.IGNORECASE)
    
#     # Extract the main content more carefully
#     # If "telling the benefits of" pattern exists, preserve it
#     if re.search(r'telling\s+the\s+benefits\s+of', cleaned, flags=re.IGNORECASE):
#         main_content = re.search(r'telling\s+the\s+benefits\s+of\s+(.+?)(?:\.|$)', cleaned, flags=re.IGNORECASE)
#         if main_content:
#             return f"the benefits of {main_content.group(1)}".strip()
    
#     # Remove "to [Name]" patterns
#     cleaned = re.sub(r'to\s+([A-Z][a-z]+)(?:\s|,|\.)', '', cleaned, flags=re.IGNORECASE)
    
#     # Remove email address references
#     cleaned = re.sub(r'(?:his|her|their)\s+email\s+(?:is|address\s+is)', '', cleaned, flags=re.IGNORECASE)
    
#     # Remove asking/telling mentions but preserve what comes after
#     telling_match = re.search(r'(?:asking|telling)\s+(?:him|her|them)\s+(.+)', cleaned, flags=re.IGNORECASE)
#     if telling_match:
#         return telling_match.group(1).strip()
    
#     return cleaned.strip()

# def extract_project_name(query):
#     """Extract project name from query if available."""
#     match = re.search(r'(?:project|task)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     return match.group(1).strip() if match else None

# def extract_benefit_topic(query):
#     """Extract topic for benefits discussion."""
#     match = re.search(r'benefits\s+of\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     return match.group(1).strip() if match else None

# def extract_recipient_name(email):
#     """Extract recipient name from email address."""
#     name = email.split('@')[0]
#     # Convert email format to proper name (e.g., john.doe -> John Doe)
#     if '.' in name:
#         parts = name.split('.')
#         return ' '.join(part.capitalize() for part in parts)
#     return name.capitalize()

# def generate_subject(query):
#     """Generate appropriate subject line based on query content."""
#     # First check for explicit subject mentions
#     subject_patterns = [
#         r'(?:subject|about|regarding|re|titled)[:|\s]\s*"?([^"\.]+)"?',
#         r'(?:email|message|send)\s+(?:with\s+subject|about|regarding)\s+"?([^"\.]+)"?',
#         r'with\s+(?:subject|title)\s+"?([^"\.]+)"?'
#     ]
    
#     for pattern in subject_patterns:
#         subject_match = re.search(pattern, query, re.IGNORECASE)
#         if subject_match:
#             return subject_match.group(1).strip()
    
#     # Check for benefits of something
#     benefits_match = re.search(r'benefits\s+of\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     if benefits_match:
#         return f"Benefits of {benefits_match.group(1).capitalize()}"
    
#     # Look for project mentions
#     project_match = re.search(r'(?:project|task)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
#     if project_match:
#         project = project_match.group(1).strip()
#         # Check if this is an update request
#         if re.search(r'update|progress|status', query, re.IGNORECASE):
#             return f"Update Request: Project {project}"
#         return f"Regarding Project {project}"
    
#     # Check for "telling" pattern
#     telling_match = re.search(r'telling\s+(?:him|her|them)\s+(?:about|regarding)?\s*(.+?)(?:\.|$)', query, re.IGNORECASE)
#     if telling_match:
#         topic = telling_match.group(1).strip()
#         return f"Information about {topic}"
    
#     # Check for update requests
#     if re.search(r'update|progress|status', query, re.IGNORECASE):
#         return "Project Update Request"
    
#     # Generic subject that uses the first few words of content
#     content = extract_main_content(query)
#     words = content.split()[:3]  # First 3 words
#     if words:
#         return " ".join(words).capitalize()
#     return "Information Request"

# def generate_email_content(recipient_name, subject, query):
#     """Generate professional email content based on query type and context."""
#     content = extract_main_content(query)
    
#     # Standard professional email template
#     body = [
#         f"Hello {recipient_name},",
#         "",
#         "I hope this email finds you well."
#     ]
    
#     # Determine email type and generate appropriate content
#     if "access" in query.lower() and "software" in query.lower():
#         body.extend([
#             "",
#             "I am writing to request access to the new software. I would need this access to perform my duties effectively.",
#             "",
#             "Could you please provide me with the necessary credentials or instructions to access the system?"
#         ])
#     elif "update" in query.lower() or "status" in query.lower() or "progress" in query.lower():
#         project = extract_project_name(query) or "the project"
#         body.extend([
#             "",
#             f"I am writing to request an update on {project}.",
#             "",
#             "Could you please share with me:",
#             "- Current progress and status",
#             "- Any challenges or blockers faced",
#             "- Expected timeline for completion",
#             "- Any additional resources needed"
#         ])
#     elif "meeting" in query.lower() or "schedule" in query.lower():
#         body.extend([
#             "",
#             "I would like to schedule a meeting to discuss our current projects and priorities.",
#             "",
#             "Please let me know your availability for next week so we can find a suitable time.",
#             "",
#             "Agenda items for discussion:",
#             "- Project updates",
#             "- Timeline review",
#             "- Resource allocation",
#             "- Next steps"
#         ])
#     elif "document" in query.lower() or "file" in query.lower():
#         body.extend([
#             "",
#             "I am writing to request the documents related to our recent project.",
#             "",
#             "Could you please share these files with me at your earliest convenience?",
#             "",
#             "This will help ensure we maintain proper documentation and can proceed efficiently."
#         ])
#     elif "benefits" in query.lower():
#         topic = extract_benefit_topic(query) or "the program"
#         body.extend([
#             "",
#             f"I wanted to share with you some key benefits of {topic}:",
#             "",
#             "- Improved efficiency and productivity",
#             "- Cost savings and resource optimization",
#             "- Enhanced collaboration and communication",
#             "- Better outcomes and quality results",
#             "- Long-term sustainability and scalability",
#             "",
#             f"I believe implementing {topic} would be valuable for our team and organization."
#         ])
#     else:
#         body.extend([
#             "",
#             f"{content}",
#             "",
#             "I look forward to your insights on this matter."
#         ])
    
#     body.extend([
#         "",
#         "I would appreciate your response at your earliest convenience. If you need any additional information or have questions, please don't hesitate to ask.",
#         "",
#         "Best regards,",
#         "Milind Warade"
#     ])
    
#     return "\n".join(body)

# def send_email(query: str) -> str:
#     """Send an email based on the user query with improved content generation."""
#     print(f"Debug: send_email called with query: {query}")
#     try:
#         # Extract email address
#         email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
#         email_matches = re.findall(email_pattern, query)
        
#         if not email_matches:
#             return "❌ No email address found in the query! Please include a valid email address."
            
#         recipient = email_matches[0]
#         print(f"Debug: Recipient email extracted: {recipient}")
        
#         # Get Google credentials and initialize Gmail API
#         credentials = None
#         token_path = os.path.join(os.path.expanduser('~'), '.config', 'google', 'token.json')
#         credentials_path = os.path.join(os.path.expanduser('~'), '.config', 'google', 'credentials.json')
        
#         SCOPES = ['https://www.googleapis.com/auth/gmail.send',
#                   'https://www.googleapis.com/auth/gmail.compose']
        
#         try:
#             if os.path.exists(token_path):
#                 credentials = Credentials.from_authorized_user_file(token_path, SCOPES)
            
#             # If credentials don't exist or are invalid, run the OAuth flow
#             if not credentials or not credentials.valid:
#                 if credentials and credentials.expired and credentials.refresh_token:
#                     credentials.refresh(Request())
#                 else:
#                     if not os.path.exists(credentials_path):
#                         return "❌ Google credentials file not found. Please set up OAuth 2.0 credentials."
                    
#                     flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
#                     credentials = flow.run_local_server(port=0)
                
#                 # Save the credentials for future use
#                 os.makedirs(os.path.dirname(token_path), exist_ok=True)
#                 with open(token_path, 'w') as token:
#                     token.write(credentials.to_json())
            
#             # Build Gmail service
#             gmail = build('gmail', 'v1', credentials=credentials)
            
#             # Extract recipient name and generate subject/content
#             recipient_name = extract_recipient_name(recipient)
#             subject = generate_subject(query)
#             body = generate_email_content(recipient_name, subject, query)
            
#             # Get sender email
#             sender = gmail.users().getProfile(userId='me').execute()['emailAddress']
            
#             # Create and send message
#             message = MIMEText(body)
#             message['to'] = recipient
#             message['from'] = sender
#             message['subject'] = subject
            
#             raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#             send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
            
#             return f"""✅ Email sent successfully!
# To: {recipient}
# Subject: {subject}
# Message:
# {body}"""
            
#         except Exception as e:
#             error_trace = traceback.format_exc()
#             print(f"Debug: Error in Gmail API operations: {str(e)}")
#             print(f"Debug: Traceback: {error_trace}")
#             return f"❌ Email sending failed: {str(e)}"
            
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"Debug: Error in send_email: {str(e)}")
#         print(f"Debug: Traceback: {error_trace}")
#         return f"❌ Failed to process email request: {str(e)}"

#6

def extract_main_content(query):
    """Extract the main content/purpose from the query."""
    # Remove email addresses
    cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', query)
    
    # Remove email related commands
    cleaned = re.sub(r'(?:send|write|compose|email)\s+(?:an|a)?\s*email\s+(?:to|about|regarding|on|with)?', '', cleaned, flags=re.IGNORECASE)
    
    # Remove "requesting" if it exists
    cleaned = re.sub(r'requesting\s+', '', cleaned, flags=re.IGNORECASE)
    
    # Extract the main content more carefully
    # If "telling the benefits of" pattern exists, preserve it
    if re.search(r'telling\s+the\s+benefits\s+of', cleaned, flags=re.IGNORECASE):
        main_content = re.search(r'telling\s+the\s+benefits\s+of\s+(.+?)(?:\.|$)', cleaned, flags=re.IGNORECASE)
        if main_content:
            return f"the benefits of {main_content.group(1)}".strip()
    
    # Remove "to [Name]" patterns
    cleaned = re.sub(r'to\s+([A-Z][a-z]+)(?:\s|,|\.)', '', cleaned, flags=re.IGNORECASE)
    
    # Remove email address references
    cleaned = re.sub(r'(?:his|her|their)\s+email\s+(?:is|address\s+is)', '', cleaned, flags=re.IGNORECASE)
    
    # Remove asking/telling mentions but preserve what comes after
    telling_match = re.search(r'(?:asking|telling)\s+(?:him|her|them)\s+(.+)', cleaned, flags=re.IGNORECASE)
    if telling_match:
        return telling_match.group(1).strip()
    
    return cleaned.strip()

def extract_project_name(query):
    """Extract project name from query if available."""
    match = re.search(r'(?:project|task)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_benefit_topic(query):
    """Extract topic for benefits discussion."""
    match = re.search(r'benefits\s+of\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_recipient_name(email):
    """Extract recipient name from email address."""
    name = email.split('@')[0]
    # Convert email format to proper name (e.g., john.doe -> John Doe)
    if '.' in name:
        parts = name.split('.')
        return ' '.join(part.capitalize() for part in parts)
    return name.capitalize()

def generate_subject(query):
    """Generate appropriate subject line based on query content."""
    # First check for explicit subject mentions
    subject_patterns = [
        r'(?:subject|about|regarding|re|titled)[:|\s]\s*"?([^"\.]+)"?',
        r'(?:email|message|send)\s+(?:with\s+subject|about|regarding)\s+"?([^"\.]+)"?',
        r'with\s+(?:subject|title)\s+"?([^"\.]+)"?'
    ]
    
    for pattern in subject_patterns:
        subject_match = re.search(pattern, query, re.IGNORECASE)
        if subject_match:
            return subject_match.group(1).strip()
    
    # Check for benefits of something
    benefits_match = re.search(r'benefits\s+of\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
    if benefits_match:
        return f"Benefits of {benefits_match.group(1).capitalize()}"
    
    # Look for project mentions
    project_match = re.search(r'(?:project|task)\s+([a-zA-Z0-9\s]+)', query, re.IGNORECASE)
    if project_match:
        project = project_match.group(1).strip()
        # Check if this is an update request
        if re.search(r'update|progress|status', query, re.IGNORECASE):
            return f"Update Request: Project {project}"
        return f"Regarding Project {project}"
    
    # Check for "telling" pattern
    telling_match = re.search(r'telling\s+(?:him|her|them)\s+(?:about|regarding)?\s*(.+?)(?:\.|$)', query, re.IGNORECASE)
    if telling_match:
        topic = telling_match.group(1).strip()
        return f"Information about {topic}"
    
    # Check for update requests
    if re.search(r'update|progress|status', query, re.IGNORECASE):
        return "Project Update Request"
    
    # Generic subject that uses the first few words of content
    content = extract_main_content(query)
    words = content.split()[:3]  # First 3 words
    if words:
        return " ".join(words).capitalize()
    return "Information Request"

def generate_email_content(recipient_name, subject, query):
    """Generate professional email content based on query type and context."""
    content = extract_main_content(query)
    
    # Standard professional email template
    body = [
        f"Hello {recipient_name},",
        "",
        "I hope this email finds you well."
    ]
    
    # Determine email type and generate appropriate content
    if "access" in query.lower() and "software" in query.lower():
        body.extend([
            "",
            "I am writing to request access to the new software. I would need this access to perform my duties effectively.",
            "",
            "Could you please provide me with the necessary credentials or instructions to access the system?"
        ])
    elif "update" in query.lower() or "status" in query.lower() or "progress" in query.lower():
        project = extract_project_name(query) or "the project"
        body.extend([
            "",
            f"I am writing to request an update on {project}.",
            "",
            "Could you please share with me:",
            "- Current progress and status",
            "- Any challenges or blockers faced",
            "- Expected timeline for completion",
            "- Any additional resources needed"
        ])
    elif "meeting" in query.lower() or "schedule" in query.lower():
        body.extend([
            "",
            "I would like to schedule a meeting to discuss our current projects and priorities.",
            "",
            "Please let me know your availability for next week so we can find a suitable time.",
            "",
            "Agenda items for discussion:",
            "- Project updates",
            "- Timeline review",
            "- Resource allocation",
            "- Next steps"
        ])
    elif "document" in query.lower() or "file" in query.lower():
        body.extend([
            "",
            "I am writing to request the documents related to our recent project.",
            "",
            "Could you please share these files with me at your earliest convenience?",
            "",
            "This will help ensure we maintain proper documentation and can proceed efficiently."
        ])
    elif "benefits" in query.lower():
        topic = extract_benefit_topic(query) or "the program"
        body.extend([
            "",
            f"I wanted to share with you some key benefits of {topic}:",
            "",
            "- Improved efficiency and productivity",
            "- Cost savings and resource optimization",
            "- Enhanced collaboration and communication",
            "- Better outcomes and quality results",
            "- Long-term sustainability and scalability",
            "",
            f"I believe implementing {topic} would be valuable for our team and organization."
        ])
    else:
        body.extend([
            "",
            f"{content}",
            "",
            "I look forward to your insights on this matter."
        ])
    
    body.extend([
        "",
        "I would appreciate your response at your earliest convenience. If you need any additional information or have questions, please don't hesitate to ask.",
        "",
        "Best regards,",
        "Milind Warade"
    ])
    
    return "\n".join(body)

def send_email(query: str) -> str:
    """Send an email based on the user query with improved content generation."""
    print(f"Debug: send_email called with query: {query}")
    try:
        # Get credentials path directly
        credentials_path = get_google_credentials()
        print(f"Debug: Using credentials from: {credentials_path}")
        
        if not credentials_path:
            print("Debug: No valid credentials found")
            return "❌ Google credentials not available. Unable to send email."
            
        # Extract email address
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        email_matches = re.findall(email_pattern, query)
        
        if not email_matches:
            return "❌ No email address found in the query! Please include a valid email address."
            
        recipient = email_matches[0]
        print(f"Debug: Recipient email extracted: {recipient}")
        
        # Import Google libraries only when needed
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from email.mime.text import MIMEText
        import base64
        
        # Initialize Gmail service directly with credentials
        creds = Credentials.from_authorized_user_file(credentials_path, 
            ['https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.send'])
        gmail = build('gmail', 'v1', credentials=creds)
        print("Debug: Gmail service built successfully")
        
        # Extract recipient name and generate subject/content
        recipient_name = extract_recipient_name(recipient)
        subject = generate_subject(query)
        body = generate_email_content(recipient_name, subject, query)
        
        try:
            # Get sender email
            profile_response = gmail.users().getProfile(userId='me').execute()
            sender = profile_response['emailAddress']
            print(f"Debug: Sender email retrieved: {sender}")
            
            # Create and send message
            message = MIMEText(body)
            message['to'] = recipient
            message['from'] = sender
            message['subject'] = subject
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            send_result = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
            print(f"Debug: Email sent successfully, message ID: {send_result.get('id', 'unknown')}")
            
            return f"""✅ Email sent successfully!
To: {recipient}
Subject: {subject}
Message:
{body}"""
            
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Debug: Error sending email: {str(e)}")
            print(f"Debug: Traceback: {error_trace}")
            return f"❌ Email sending failed: {str(e)}"
            
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Debug: Error in send_email: {str(e)}")
        print(f"Debug: Traceback: {error_trace}")
        return f"❌ Failed to process email request: {str(e)}"



# Agent Manager Functions
def route_query(state: AgentState) -> AgentState:
    """Route the query to the appropriate action based on keywords or LLM."""
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
    
    # Advanced pattern matching for more specific cases
    if re.search(r'(meeting|appointment|call|interview)\s+(?:with|on)\s+', query, re.IGNORECASE):
        state["actions"] = ["calendar_create"]
        return state
        
    if re.search(r'what.*calendar', query, re.IGNORECASE) or re.search(r'(today|tomorrow|this week).*events', query, re.IGNORECASE):
        state["actions"] = ["calendar_list"]
        return state
        
    if re.search(r'(contact|message|reach out to)\s+.+@.+\..+', query, re.IGNORECASE):
        state["actions"] = ["email"]
        return state
    
    # Fallback to LLM for more complex queries
    groq_client = get_groq_client()
    if groq_client:
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
    
    # Default fallback - try to guess from keywords
    if "schedule" in query.lower() or "meeting" in query.lower() or "event" in query.lower():
        state["actions"] = ["calendar_create"]
    elif "email" in query.lower() or "@" in query:
        state["actions"] = ["email"]
    else:
        state["actions"] = ["calendar_list"]  # Default fallback
        
    return state

def execute_tools(state: AgentState) -> AgentState:
    """Execute the appropriate actions based on the routing."""
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
            traceback.print_exc()
    
    if not responses:
        state["final_response"] = "I couldn't process your request. Please try again."
    else:
        state["final_response"] = "\n\n".join(responses)
    
    return state

def agent_manager(query: str) -> str:
    """Create and execute the agent workflow."""
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

# Cache initialization functions
@st.cache_resource
def initialize_apis():
    """Initialize API clients with caching to avoid repeated initialization."""
    groq = get_groq_client()
    credentials = get_google_credentials()
    return {"groq": groq, "google_credentials": credentials}

# Initialize cached resources
apis = initialize_apis()

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