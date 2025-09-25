import os

from strands import tool

from rllm.tools.web_tools.gsearch_tool import GoogleSearchTool

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Create an instance of the existing GoogleSearchTool
gsearch_instance = GoogleSearchTool()


@tool
def google_search(query: str) -> str:
    """
    Search a query using the Google search engine, returning the top results along with a short snippet about their contents.

    Args:
        query: Query to be submitted to Google search engine.

    Returns:
        Search results as a formatted string.
    """
    try:
        # Check if environment variables are set
        if not os.getenv("GOOGLE_SEARCH_SECRET_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
            return "Error: Google Search API credentials not configured. Please set GOOGLE_SEARCH_SECRET_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables."

        # Use the existing tool's forward method
        result = gsearch_instance.forward(query)

        # Check for errors in the ToolOutput
        if hasattr(result, "error") and result.error:
            return f"Error: {result.error}"

        # Check if we have output
        if not hasattr(result, "output") or not result.output:
            # This means the API call failed or returned no results
            # Let's provide a more helpful error message
            return "Search failed: The Google Search API returned an error or no results. Please check your API credentials and try again."

        # Format the results nicely
        formatted_results = []
        for i, (link, snippet) in enumerate(result.output.items(), 1):
            formatted_results.append(f"{i}. {link}\n   {snippet}\n")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error: {type(e).__name__} - {str(e)}"


if __name__ == "__main__":
    # Test the tool
    result = google_search("Python programming")
    print(result)
