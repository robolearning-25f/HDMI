import pandas as pd
import numpy as np
import plotly.graph_objects as go

# pip install plotly kaleido
csv_path = "point_errors.csv"
df = pd.read_csv(csv_path)

steps = df["_step"].values

point_styles = {
    0: {"line": "rgb(31, 119, 180)", "fill": "rgba(31, 119, 180, 0.30)"},
    1: {"line": "rgb(214, 39, 40)",  "fill": "rgba(214, 39, 40, 0.30)"},
}

def make_fig(dim: int):
    fig = go.Figure()

    if dim == 2:
        unit_label = "px"
        scale = 1.0
    else:
        unit_label = "cm"
        scale = 100.0  # meters -> cm

    for point in [0, 1]:
        style = point_styles[point]
        line_color = style["line"]
        fill_color = style["fill"]

        # per-env faint lines (just context)
        for env in range(8):
            col = f"env_{env}/point_{point}_error_{dim}d"
            vals = df[col].to_numpy() * scale

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=vals,
                    mode="lines",
                    name=f"env_{env} point {point} ({dim}D)",
                    line=dict(width=1, color=line_color),
                    opacity=0.08,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # mean / std across envs
        cols = [f"env_{env}/point_{point}_error_{dim}d" for env in range(8)]
        values = df[cols].to_numpy() * scale
        mean = values.mean(axis=1)
        std = values.std(axis=1)

        upper = mean + std
        lower = mean - std

        # variance band (stronger color, visible)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([steps, steps[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                hoverinfo="skip",
                name=f"Point {point} ±1σ ({dim}D)",
                showlegend=True,
            )
        )

        # upper / lower boundary lines to emphasize variance
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=upper,
                mode="lines",
                line=dict(width=1.5, color=line_color, dash="dot"),
                name=f"Point {point} +1σ ({dim}D)",
                showlegend=False,
                hovertemplate=f"step=%{{x}}<br>+1σ=%{{y:.4f}} {unit_label}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=lower,
                mode="lines",
                line=dict(width=1.5, color=line_color, dash="dot"),
                name=f"Point {point} -1σ ({dim}D)",
                showlegend=False,
                hovertemplate=f"step=%{{x}}<br>-1σ=%{{y:.4f}} {unit_label}<extra></extra>",
            )
        )

        # mean line on top
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines+markers",
                name=f"Point {point} mean ({dim}D)",
                line=dict(width=3, color=line_color),
                marker=dict(size=5),
                hovertemplate=f"step=%{{x}}<br>mean=%{{y:.4f}} {unit_label}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{dim}D Contact Point Error vs Step",
        xaxis_title="Step",
        yaxis_title=f"{dim}D error ({unit_label})",
        template="plotly_white",
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=80, r=20, t=80, b=60),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


fig_2d = make_fig(2)
fig_3d = make_fig(3)

fig_2d.write_image("point_error_2d_fancy.png", scale=3)
fig_3d.write_image("point_error_3d_fancy.png", scale=3)

# Calculate overall 2D and 3D errors
print("\n" + "="*50)
print("OVERALL ERROR STATISTICS")
print("="*50)

# 2D errors (in pixels)
error_2d_cols = [f"env_{env}/point_{point}_error_2d" for env in range(8) for point in [0, 1]]
all_2d_errors = df[error_2d_cols].values.flatten()
error_2d_mean = np.mean(all_2d_errors)
error_2d_std = np.std(all_2d_errors)

print(f"\n2D Error (pixels):")
print(f"  Mean: {error_2d_mean:.4f} px")
print(f"  Std:  {error_2d_std:.4f} px")

# 3D errors (in meters, displayed as cm)
error_3d_cols = [f"env_{env}/point_{point}_error_3d" for env in range(8) for point in [0, 1]]
all_3d_errors = df[error_3d_cols].values.flatten()
error_3d_mean = np.mean(all_3d_errors) * 100  # convert to cm
error_3d_std = np.std(all_3d_errors) * 100    # convert to cm

print(f"\n3D Error (centimeters):")
print(f"  Mean: {error_3d_mean:.4f} cm")
print(f"  Std:  {error_3d_std:.4f} cm")
print("\n" + "="*50)
