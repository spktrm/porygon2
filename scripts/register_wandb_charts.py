"""One-off: register the custom Vega-Lite payoff-heatmap chart preset with
the wandb backend (entity-scoped, created once via the API — there is no
programmatic "create panel" any other way; the UI's Vega editor is the only
alternative). Run this whenever the spec below changes; it creates a NEW
preset id each time (names must be unique), so after editing the spec, bump
_CHART_NAME and update the id learner.py / wandb_views.py reference.

    env/bin/python scripts/register_wandb_charts.py
"""

import wandb

_ENTITY = "jtwin"
_CHART_NAME = "league-payoff-heatmap-v10"
_DISPLAY_NAME = "League payoff heatmap"


def _winrate_hex(winrate: float) -> str:
    """Red/gold/green hex for a win rate. Kept in sync by hand with
    Learner._winrate_hex in rl/online/learner.py — dead code there now
    (see below), duplicated here only because this script has no jax/model
    deps and shouldn't import learner.py just for this."""
    red, gold, green = (211, 0, 0), (255, 205, 50), (50, 205, 50)
    if winrate <= 0.25:
        r, g, b = red
    elif winrate <= 0.5:
        t = (winrate - 0.25) / 0.25
        r, g, b = (round(a + (c - a) * t) for a, c in zip(red, gold))
    elif winrate <= 0.75:
        t = (winrate - 0.5) / 0.25
        r, g, b = (round(a + (c - a) * t) for a, c in zip(gold, green))
    else:
        r, g, b = green
    return f"#{r:02x}{g:02x}{b:02x}"


# v3-v6 (explicit color.scale.range hex array), v7 (color.scale.scheme +
# domain + clamp), v8 (per-cell literal hex column + scale: null), and v9
# (v8 with the template field key renamed off "color") were all confirmed
# spec-correct -- via wandb's own GraphQL API re-fetching the stored spec
# byte-for-byte, and via a neutral standalone Vega-Lite renderer
# (vl-convert) producing exactly the intended red -> gold -> green -- yet
# every one rendered as either an unrelated pink/black/blue palette or a
# single flat colour in wandb's actual custom-chart panel (confirmed via
# the downloaded panel SVG: every cell baked in with the identical literal
# fill regardless of field name or data values). The common factor across
# every failing version: the rect mark's fill was bound to a table FIELD
# via "field" in the color encoding. The one encoding that DID render
# correctly the whole time was the text mark's black/white choice, which
# uses "condition"/"value" with NO "field" at all -- pure literal values
# selected by a boolean test. v10 applies that same pattern to the rect:
# a chain of "condition" tests against the (quantitative, field-bound)
# winrate value picking a literal hex "value" per 5%-wide band, and a
# fallback "value" for the top band. No field is ever bound directly to
# color/fill -- only used inside test expressions -- which is the one
# combination not yet tried.
_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
_COLOR_CONDITIONS = [
    {
        "test": f"datum['${{field:winrate}}'] <= {t}",
        "value": _winrate_hex(t - 0.025),
    }
    for t in _THRESHOLDS
]
_COLOR_FALLBACK = _winrate_hex(1.0)

_SPEC = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": "${string:title}",
    "data": {"name": "wandb"},
    "width": "container",
    "height": "container",
    "autosize": {"type": "fit", "contains": "padding"},
    "config": {
        "axis": {"grid": False},
        "view": {"stroke": None},
        "mark": {"opacity": 1, "fillOpacity": 1},
    },
    "layer": [
        {
            "mark": {
                "type": "rect",
                "stroke": "white",
                "strokeWidth": 0.5,
                "opacity": 1,
                "fillOpacity": 1,
            },
            "encoding": {
                "x": {
                    "field": "${field:col}",
                    "type": "ordinal",
                    "sort": {"field": "${field:col_idx}"},
                    "axis": {
                        "title": "opponent",
                        "labelAngle": -40,
                        "labelAlign": "right",
                    },
                },
                "y": {
                    "field": "${field:row}",
                    "type": "ordinal",
                    "sort": {"field": "${field:row_idx}"},
                    "axis": {"title": "player"},
                },
                "color": {
                    "condition": _COLOR_CONDITIONS,
                    "value": _COLOR_FALLBACK,
                },
                "tooltip": [
                    {"field": "${field:row}", "type": "nominal", "title": "player"},
                    {"field": "${field:col}", "type": "nominal", "title": "opponent"},
                    {
                        "field": "${field:winrate}",
                        "type": "quantitative",
                        "title": "win rate",
                        "format": ".1%",
                    },
                ],
            },
        },
        {
            "mark": {"type": "text", "fontSize": 11},
            "encoding": {
                "x": {
                    "field": "${field:col}",
                    "type": "ordinal",
                    "sort": {"field": "${field:col_idx}"},
                },
                "y": {
                    "field": "${field:row}",
                    "type": "ordinal",
                    "sort": {"field": "${field:row_idx}"},
                },
                "text": {
                    "field": "${field:winrate}",
                    "type": "quantitative",
                    "format": ".0%",
                },
                "color": {
                    "condition": {
                        "test": (
                            "datum['${field:winrate}'] > 0.7 || "
                            "datum['${field:winrate}'] < 0.3"
                        ),
                        "value": "white",
                    },
                    "value": "black",
                },
            },
        },
    ],
}


def main():
    api = wandb.Api()
    chart_id = api.create_custom_chart(
        entity=_ENTITY,
        name=_CHART_NAME,
        display_name=_DISPLAY_NAME,
        spec_type="vega2",
        access="private",
        spec=_SPEC,
    )
    print(chart_id)


if __name__ == "__main__":
    main()
