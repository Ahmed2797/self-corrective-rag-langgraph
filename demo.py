from src.pipeline import pipeline

app = pipeline()
app

def print_graph(app):
    """
    Print the LangGraph workflow as an ASCII diagram.
    """
    print("\n" + "=" * 80)
    print("LANGGRAPH WORKFLOW")
    print("=" * 80)

    # Terminal visualization
    # print(app.get_graph().draw_ascii())

    # Mermaid source when needed
    # print(app.get_graph().draw_mermaid())
    graph = app.get_graph()

    png_bytes = graph.draw_mermaid_png()

    with open("langgraph_workflow.png", "wb") as f:
        f.write(png_bytes)

    print("Graph saved to langgraph_workflow.png")

    print("=" * 80)


print_graph(app)

### pip install grandalf