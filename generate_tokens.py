import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Define the scopes for Google Calendar and Gmail APIs
SCOPES = [
    'https://www.googleapis.com/auth/calendar.events',  # For creating and reading events
    'https://www.googleapis.com/auth/calendar.readonly',  # For reading calendar events
    'https://www.googleapis.com/auth/gmail.compose',  # For drafting emails
    'https://www.googleapis.com/auth/gmail.send'  # For sending emails
]

def get_authenticated_service():
    """
    Authenticate and get credentials for Google Calendar and Gmail APIs.

    Returns:
    - Authenticated credentials
    - None if authentication fails
    """
    creds = None
    
    # Token file path
    token_path = 'token.json'
    
    # Client secret file path - updated to match your renamed file
    client_secret_path = 'client_secrets.json'
    
    # Check if token file exists
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                creds = None
        
        # If no valid credentials, initiate OAuth flow
        if not creds:
            try:
                # Allow OAuth for local testing
                os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_secret_path, SCOPES)
                creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                    
                # Also save in the format your application expects
                token_info = {
                    'token': creds.token,
                    'refresh_token': creds.refresh_token,
                    'token_uri': creds.token_uri,
                    'client_id': creds.client_id,
                    'client_secret': creds.client_secret,
                    'scopes': SCOPES,
                    'universe_domain': 'googleapis.com'
                }
                
                # Save to token.json in a more readable format
                with open(token_path, 'w') as f:
                    json.dump(token_info, f, indent=4)
                    
            except Exception as e:
                print(f"Authentication failed: {e}")
                return None
    
    return creds

def main():
    """
    Main function to execute the authentication process.
    """
    print("Starting authentication process...")
    credentials = get_authenticated_service()
    if credentials:
        print("\n✅ Success! New token information has been saved to token.json")
        print("You can now update your Streamlit secrets with this information")
    else:
        print("\n❌ Authentication failed.")

if __name__ == '__main__':
    main()