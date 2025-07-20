
from universal_mcp.servers import SingleMCPServer

from universal_mcp_graph_rag.app import GraphRagApp

app_instance = GraphRagApp()

mcp = SingleMCPServer(
    app_instance=app_instance
)

if __name__ == "__main__":
    print(f"Starting {mcp.name}...")
    mcp.run()


