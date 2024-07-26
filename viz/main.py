import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def main():
    # Load the data from JSON file
    with open("viz/matchupResults.json", "r") as f:
        results = json.load(f)

    # Process the data
    data = np.array(results).reshape(9, 9, -1)
    data = data.mean(-1)
    data = (data + 1) / 2

    # Create heatmap using Plotly Express
    fig = px.imshow(data, text_auto=True, aspect="auto")

    # Show the figure
    fig.show()


if __name__ == "__main__":
    main()
