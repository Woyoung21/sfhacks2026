from linkup import LinkupClient

client = LinkupClient(
    api_key="2951c1df-bf36-40da-81a0-1d940051ab89",  # Or set the LINKUP_API_KEY environment variable
)

# Perform a search query
search_response = client.search(
    query="What is the weather in San Francisco today?",
    depth="standard",
    output_type="sourcedAnswer",
)
print(search_response.answer)

# # Fetch the content of a web page
# response = client.fetch(
#     url="https://docs.linkup.so",
# )
# print(response)
