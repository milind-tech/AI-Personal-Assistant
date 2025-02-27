import streamlit as st
# Set page configuration
st.set_page_config(
    page_title="AI Personal Assistant",
    page_icon="🤖",
    layout="wide"
)

from agent_managerr import agent_manager  
# Add custom CSS for styling input box
st.markdown("""
    <style>
    .stTextInput > div > div > input {
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

                # ✅ Ensure response is a string
                if not isinstance(response, str):
                    response = "❌ Error: No valid response received."

                # Display response
                st.markdown("### Response:")
                
                # ✅ Use Streamlit container instead of <div>
                with st.container():
                    # Split response into lines and format them
                    lines = response.split('\n')
                    for line in lines:
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

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a request first.")

# Footer
st.markdown("---")
st.markdown("Made by Milind Warade")