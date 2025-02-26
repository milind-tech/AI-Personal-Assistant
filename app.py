import streamlit as st
from agent_managerr import agent_manager  # This is the only import we need





# Set page configuration
st.set_page_config(
    page_title="AI Personal Assistant",
    page_icon="🤖",
    layout="wide"
)


# Add custom CSS
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        min-height: 100px;
    }
    .output-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
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
                
                # Display response in a container with proper formatting
                st.markdown("### Response:")
                st.markdown('<div class="output-container">', unsafe_allow_html=True)
                
                # Split response into lines and format them
                lines = response.split('\n')
                for line in lines:
                    # Add emojis and formatting based on content
                    if "Event Scheduled Successfully" in line:
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
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a request first.")

# Footer
st.markdown("---")
st.markdown("Made by Milind Warade")



# cd C:\Users\VICTUS\Desktop\AGENTS
# python -m streamlit run app.py