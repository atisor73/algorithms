import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import scipy.special

import bokeh.io
bokeh.io.output_notebook()

import panel as pn
pn.extension()

from .prior_inverse_search import *

def style(p, autohide=False):
    p.title.text_font="Helvetica"
    p.title.text_font_size="16px"
    p.title.align="center"
    p.xaxis.axis_label_text_font="Helvetica"
    p.yaxis.axis_label_text_font="Helvetica"

    p.xaxis.axis_label_text_font_size="13px"
    p.yaxis.axis_label_text_font_size="13px"
#     p.xaxis.axis_label_text_font_style = "normal"
#     p.yaxis.axis_label_text_font_style = "normal"
    p.background_fill_alpha = 0
    if autohide: p.toolbar.autohide=True
    return p

# COLORS
purple = "#aa6abe"
blue = "#5499c7"
orange = "#E67E22"
green = "#52be80"
pink = "#DD7295"
red = "#cd6155"

color_normal = purple
color_studentt = blue
color_exponential = orange
color_gamma = green
color_invgamma = "orange"
color_weibull = "teal"
color_pareto = pink
color_lognormal = "#7162EE"
color_cauchy = "#C7545F"
color_gumbel = "#B5A642"

# MISCELLANEOUS PLOTTING FUNCTIONS
def _pdfcdf_2p2b_plotter(
    L, U, mu, sigma, f_pdf, f_cdf,
    x_patch, pdf_patch, cdf_patch,
    x_full, pdf_full, cdf_full,
    color="purple"
):
    """
    # 2p2b: 2 parameters (function call)
            2 boundaries (2 circles)
    """
    p_pdf = bokeh.plotting.figure(title="pdf", width=350, height=220)
    p_cdf = bokeh.plotting.figure(title="cdf", width=350, height=220)

    p_pdf.patch(x_patch, pdf_patch, color='#eaeaea')
    p_cdf.patch(x_patch, cdf_patch, color='#eaeaea')

    p_pdf.line(x_full, pdf_full, line_width=3.5, color=color)
    p_cdf.line(x_full, cdf_full, line_width=3.5, color=color)

    p_pdf.circle(L, f_pdf(L, mu, sigma), size=9, line_width=2.5, line_color=color, fill_color="white")
    p_pdf.circle(U, f_pdf(U, mu, sigma), size=9, line_width=2.5, line_color=color, fill_color="white")
    p_cdf.circle(L, f_cdf(L, mu, sigma), size=9, line_width=2.5, line_color=color, fill_color="white")
    p_cdf.circle(U, f_cdf(U, mu, sigma), size=9, line_width=2.5, line_color=color, fill_color="white")

    row_pdfcdf = pn.Row(style(p_pdf, autohide=True), style(p_cdf, autohide=True))

    return row_pdfcdf

# 1p1b: 1 parameter (function call), 1 boundary (1 circle)
def _pdfcdf_1p1b_plotter(
    U, beta, f_pdf, f_cdf,
    x_patch, pdf_patch, cdf_patch,
    x_full, pdf_full, cdf_full,
    color="green"
):
    """
    # 1p1b: 1 parameter (function call)
            1 boundary (1 circles)
    """
    p_pdf = bokeh.plotting.figure(title="pdf", width=350, height=220)
    p_cdf = bokeh.plotting.figure(title="cdf", width=350, height=220)

    p_pdf.patch(x_patch, pdf_patch, color='#eaeaea')
    p_cdf.patch(x_patch, cdf_patch, color='#eaeaea')

    p_pdf.line(x_full, pdf_full, line_width=3.5, color=color)
    p_cdf.line(x_full, cdf_full, line_width=3.5, color=color)

    p_pdf.circle(U, f_pdf(U, beta), size=9, line_width=2.5, line_color=color, fill_color="white")
    p_cdf.circle(U, f_cdf(U, beta), size=9, line_width=2.5, line_color=color, fill_color="white")

    row_pdfcdf = pn.Row(style(p_pdf, autohide=True), style(p_cdf, autohide=True))

    return row_pdfcdf


# NORMAL
half_checkbox_normal = pn.widgets.Checkbox(name='half', width=50, value=False)
L_input_normal = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_normal = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_normal = pn.widgets.FloatSlider(name="bulk %", value=99, start=50, end=99, width=150, step=1)

@pn.depends(half_checkbox_normal.param.value, watch=True)
def invisible_L_normal(half):
    if half:
        L_input_normal.value = "0"
        L_input_normal.disabled = True
        # L_input_normal.visible = False
    else:
        L_input_normal.disabled = False
        # L_input_normal.visible=True

@pn.depends(L_input_normal.param.value, U_input_normal.param.value,
            bulk_slider_normal.param.value, half_checkbox_normal.param.value)
def normal_table(L, U, bulk, half):
    L, U, bulk = float(L), float(U), float(bulk)

    if half:
        leftover = 100 - bulk
        bulk = 100 - 2*leftover
        μ, σ = find_normal(-U, U, bulk/100, precision=10)
    else:
        μ, σ = find_normal(L, U, bulk/100, precision=10)

    return pn.pane.Markdown(f"""
        | param | value |
        | ----- | ----- |
        | μ | {np.round(μ, 4)} |
        | σ | {np.round(σ, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )
@pn.depends(L_input_normal.param.value, U_input_normal.param.value,
            bulk_slider_normal.param.value, half_checkbox_normal.param.value)
def dashboard_normal(L, U, bulk, half):
    L, U, bulk = float(L), float(U), float(bulk)

    if half:
        leftover = 100 - bulk
        bulk = 100 - 2*leftover
        μ, σ = find_normal(-U, U, bulk/100, precision=10)

        f_pdf = lambda arr, mu, sigma: scipy.stats.halfnorm.pdf(arr, mu, sigma)
        f_cdf = lambda arr, mu, sigma: scipy.stats.halfnorm.cdf(arr, mu, sigma)
        padding = U * 0.3
        x = np.linspace(0, U, 1_000)
        x_high = scipy.stats.norm.ppf([0.995], μ, σ)[0]
        x_full = np.linspace(0, max(U+padding, x_high), 1_000)

        x_patch = [0] + list(x) + [U]

        pdf_full = f_pdf(x_full, μ, σ)
        cdf_full = f_cdf(x_full, μ, σ)

        pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
        cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]

        row_pdfcdf = _pdfcdf_2p2b_plotter(0, U, μ, σ, f_pdf, f_cdf,
                x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_normal)

    else:
        μ, σ = find_normal(L, U, bulk/100, precision=4)

        f_pdf = lambda arr, mu, sigma: scipy.stats.norm.pdf(arr, mu, sigma)
        f_cdf = lambda arr, mu, sigma: scipy.stats.norm.cdf(arr, mu, sigma)

        padding = (U - L) * 0.3
        x = np.linspace(L, U, 1_000)
        x_low, x_high = scipy.stats.norm.ppf([0.005, 0.995], μ, σ)
        x_full = np.linspace(min(x_low, L-padding), max(U+padding, x_high), 1_000)
        x_patch = [L] + list(x) + [U]

        pdf_full = f_pdf(x_full, μ, σ)
        cdf_full = f_cdf(x_full, μ, σ)

        pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
        cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]

        row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, μ, σ, f_pdf, f_cdf,
                x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_normal)

    return row_pdfcdf

row_LUbulk_normal = pn.Row(L_input_normal, U_input_normal, bulk_slider_normal,
    pn.Column(pn.Spacer(height=10), half_checkbox_normal), pn.Spacer(width=11), normal_table)

layout_normal = pn.Column(row_LUbulk_normal, dashboard_normal, name="Normal")



# LOG-NORMAL
L_input_lognormal = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_lognormal = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_lognormal = pn.widgets.FloatSlider(name="bulk %", value=90, start=50, end=99, width=150, step=1)

@pn.depends(L_input_lognormal.param.value, U_input_lognormal.param.value, bulk_slider_lognormal.param.value)
def lognormal_table(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    μ, σ = find_lognormal(L, U, bulk/100, precision=10)

    return pn.pane.Markdown(f"""
        | param | value |
        | ----- | ----- |
        | μ | {np.round(μ, 4)} |
        | σ | {np.round(σ, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )
@pn.depends(L_input_lognormal.param.value, U_input_lognormal.param.value, bulk_slider_lognormal.param.value)
def dashboard_lognormal(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    μ, σ = find_lognormal(L, U, bulk/100, precision=4)

    f_pdf = lambda arr, mu, sigma: scipy.stats.lognorm.pdf(arr, sigma, loc=0, scale=np.exp(mu))
    f_cdf = lambda arr, mu, sigma: scipy.stats.lognorm.cdf(arr, sigma, loc=0, scale=np.exp(mu))

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.lognorm.ppf([0.05, 0.95], σ, loc=0, scale=np.exp(μ))
    x_full = np.linspace(0, max(x_high, U+padding), 1_000)
    # x_full = np.linspace(min(L-padding,x_low), max(x_high, U+padding), 1_000)
    pdf_full = f_pdf(x_full, μ, σ)
    cdf_full = f_cdf(x_full, μ, σ)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
    cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, μ, σ, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_lognormal)

    return row_pdfcdf

row_LUbulk_lognormal = pn.Row(L_input_lognormal, U_input_lognormal,
                           bulk_slider_lognormal, pn.Spacer(width=80), lognormal_table)
layout_lognormal = pn.Column(row_LUbulk_lognormal, dashboard_lognormal, name="LogNormal")



# GAMMA
L_input_gamma = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_gamma = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_gamma = pn.widgets.FloatSlider(name="bulk %", value=99, start=50, end=99, width=150, step=1, value_throttled=True)

@pn.depends(L_input_gamma.param.value, U_input_gamma.param.value, bulk_slider_gamma.param.value)
def gamma_table(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    try:
        α, β = find_gamma(L, U, bulk=bulk/100, precision=10)
        return pn.pane.Markdown(f"""
            | param | value |
            | ----- | ----- |
            | α | {np.round(α, 4)} |
            | β | {np.round(β, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | ----- | ----- |
            | α |   |
            | β |   |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
@pn.depends(L_input_gamma.param.value, U_input_gamma.param.value, bulk_slider_gamma.param.value)
def dashboard_gamma(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    try:
        α, β = find_gamma(L, U, bulk=bulk/100, precision=4)
    except:
        α, β = 1, 1

    f_pdf = lambda arr, alpha, beta: scipy.stats.gamma.pdf(arr, alpha, scale=1/beta)
    f_cdf = lambda arr, alpha, beta: scipy.stats.gamma.cdf(arr, alpha, scale=1/beta)

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.gamma.ppf([0.005, 0.995], α, scale=1/β)
    # x_full = np.linspace(min(x_low, L-padding), max(x_high, U+padding), 1_000)
    x_full = np.linspace(0, max(x_high, U+padding), 1_000)

    pdf_full = f_pdf(x_full, α, β)
    cdf_full = f_cdf(x_full, α, β)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, α, β)) + [0]
    cdf_patch = [0] + list(f_cdf(x, α, β)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, α, β, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color_gamma)

    return row_pdfcdf

row_LUbulk_gamma = pn.Row(L_input_gamma, U_input_gamma,
                           bulk_slider_gamma, pn.Spacer(width=80), gamma_table)
layout_gamma = pn.Column(row_LUbulk_gamma, dashboard_gamma, name="Gamma")




# INVERSE GAMMA
L_input_invgamma = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_invgamma = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_invgamma = pn.widgets.FloatSlider(name="bulk %", value=95, start=50, end=99, width=150, step=1, value_throttled=True)

@pn.depends(L_input_invgamma.param.value, U_input_invgamma.param.value, bulk_slider_invgamma.param.value)
def invgamma_table(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    α, β = find_invgamma(L, U, bulk=bulk/100, precision=10)

    return pn.pane.Markdown(f"""
        | param | value |
        | ----- | ----- |
        | α | {np.round(α, 4)} |
        | β | {np.round(β, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )
@pn.depends(L_input_invgamma.param.value, U_input_invgamma.param.value, bulk_slider_invgamma.param.value)
def dashboard_invgamma(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    α, β = find_invgamma(L, U, bulk=bulk/100, precision=4)

    f_pdf = lambda arr, alpha, beta: scipy.stats.invgamma.pdf(arr, alpha, scale=beta)
    f_cdf = lambda arr, alpha, beta: scipy.stats.invgamma.cdf(arr, alpha, scale=beta)

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.invgamma.ppf([0.025, 0.975], α, scale=β)
    x_full = np.linspace(0, max(x_high, U+padding), 1_000)
    # x_full = np.linspace(min(x_low, L-padding), max(x_high, U+padding), 1_000)
    pdf_full = f_pdf(x_full, α, β)
    cdf_full = f_cdf(x_full, α, β)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, α, β)) + [0]
    cdf_patch = [0] + list(f_cdf(x, α, β)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, α, β, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color_invgamma)

    return row_pdfcdf

row_LUbulk_invgamma = pn.Row(L_input_invgamma, U_input_invgamma,
                           bulk_slider_invgamma, pn.Spacer(width=80), invgamma_table)
layout_invgamma = pn.Column(row_LUbulk_invgamma, dashboard_invgamma, name="InvGamma")




# WEIBULL
L_input_weibull = pn.widgets.TextInput(name="L", value="0.1", width=130)
U_input_weibull = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_weibull = pn.widgets.FloatSlider(name="bulk %", value=99, start=50, end=99, width=150, step=1)

@pn.depends(L_input_weibull.param.value, U_input_weibull.param.value, bulk_slider_weibull.param.value)
def weibull_table(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    α, σ = find_weibull(L, U, bulk=bulk/100, precision=10)

    return pn.pane.Markdown(f"""
        | param | value |
        | ----- | ----- |
        | α | {np.round(α, 4)} |
        | σ | {np.round(σ, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )
@pn.depends(L_input_weibull.param.value, U_input_weibull.param.value, bulk_slider_weibull.param.value)
def dashboard_weibull(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    α, σ = find_weibull(L, U, bulk=bulk/100, precision=4)

    f_pdf = lambda arr, alpha, sigma: scipy.stats.weibull_min.pdf(arr, alpha, scale=sigma)
    f_cdf = lambda arr, alpha, sigma: scipy.stats.weibull_min.cdf(arr, alpha, scale=sigma)

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.weibull_min.ppf([0.05, 0.95], α, scale=σ)
    x_full = np.linspace(0, max(x_high, U+padding), 1_000)
    # x_full = np.linspace(min(L-padding, x_low), max(x_high, U+padding), 1_000)

    pdf_full = f_pdf(x_full, α, σ)
    cdf_full = f_cdf(x_full, α, σ)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, α, σ)) + [0]
    cdf_patch = [0] + list(f_cdf(x, α, σ)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, α, σ, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color_weibull)

    return row_pdfcdf

row_LUbulk_weibull = pn.Row(L_input_weibull, U_input_weibull,
                           bulk_slider_weibull, pn.Spacer(width=80), weibull_table)
layout_weibull = pn.Column(row_LUbulk_weibull, dashboard_weibull, name="Weibull")



# EXPONENTIAL
U_input_expon = pn.widgets.TextInput(name="U", value="10", width=130)
Uppf_slider_expon = pn.widgets.FloatSlider(name="Uppf %", value=99, start=50, end=99, width=150, step=1)

@pn.depends(U_input_expon.param.value, Uppf_slider_expon.param.value)
def expon_table(U, Uppf):
    U, Uppf = float(U), float(Uppf)
    β = find_exponential(U, Uppf/100, precision=10)
    return pn.pane.Markdown(f"""
        | param | value |
        | ----------- | ----------- |
        | β | {np.round(β, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )

@pn.depends(U_input_expon.param.value, Uppf_slider_expon.param.value)
def dashboard_exponential(U, Uppf):
    U, Uppf = float(U), float(Uppf)
    β = find_exponential(U, Uppf/100, precision=10)

    f_pdf = lambda arr, beta: scipy.stats.expon.pdf(arr, loc=0, scale=1/beta)
    f_cdf = lambda arr, beta: scipy.stats.expon.cdf(arr, loc=0, scale=1/beta)

    padding = U * 0.05
    x = np.linspace(0, U, 1_000)
    x_low, x_high = scipy.stats.expon.ppf([0.005, 0.995], loc=0, scale=1/β)
    # x_full = np.linspace(-padding, x_high, 1_000)
    x_full = np.linspace(0, x_high, 1_000)

    pdf_full = f_pdf(x_full, β)
    cdf_full = f_cdf(x_full, β)

    x_patch = [0] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, β)) + [0]
    cdf_patch = [0] + list(f_cdf(x, β)) + [0]
    row_pdfcdf = _pdfcdf_1p1b_plotter(U, β, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_exponential)

    return row_pdfcdf

row_UUppf_expon = pn.Row(U_input_expon, Uppf_slider_expon, pn.Spacer(width=230), expon_table)
layout_expon = pn.Column(row_UUppf_expon, pn.Spacer(height=12), dashboard_exponential, name="Exponential")




# PARETO
ymin_input_pareto = pn.widgets.TextInput(name="ymin", value="0.1", width=70)
U_input_pareto = pn.widgets.TextInput(name="U", value="1", width=130)
Uppf_slider_pareto = pn.widgets.FloatSlider(name="Uppf %", value=99, start=50, end=99, width=150, step=1)

@pn.depends(ymin_input_pareto.param.value, U_input_pareto.param.value, Uppf_slider_pareto.param.value)
def pareto_table(ymin, U, Uppf):
    ymin, U, Uppf = float(ymin), float(U), float(Uppf)
    α = find_pareto(ymin, U, Uppf/100, precision=10)
    return pn.pane.Markdown(f"""
        | param | value |
        | ----------- | ----------- |
        | α | {np.round(α, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )

@pn.depends(ymin_input_pareto.param.value, U_input_pareto.param.value, Uppf_slider_pareto.param.value)
def dashboard_pareto(ymin, U, Uppf):
    ymin, U, Uppf = float(ymin), float(U), float(Uppf)
    α = find_pareto(ymin, U, Uppf/100, precision=10)

    f_pdf = lambda arr, alpha: scipy.stats.pareto.pdf(arr, alpha, scale=ymin)
    f_cdf = lambda arr, alpha: scipy.stats.pareto.cdf(arr, alpha, scale=ymin)

    padding = (U-ymin) * 0.1
    x = np.linspace(ymin, U, 1_000)
    x_full = np.linspace(ymin - padding, U + padding, 1_000)
    pdf_full = f_pdf(x_full, α)
    cdf_full = f_cdf(x_full, α)

    x_patch = [ymin] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, α)) + [0]
    cdf_patch = [0] + list(f_cdf(x, α)) + [0]
    row_pdfcdf = _pdfcdf_1p1b_plotter(U, α, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_pareto)

    return row_pdfcdf

row_UUppf_pareto = pn.Row(ymin_input_pareto, U_input_pareto, Uppf_slider_pareto, pn.Spacer(width=140), pareto_table)
layout_pareto = pn.Column(row_UUppf_pareto, pn.Spacer(height=12), dashboard_pareto, name="Pareto")





# CAUCHY
half_checkbox_cauchy = pn.widgets.Checkbox(name='half', width=50, value=False)
L_input_cauchy = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_cauchy = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_cauchy = pn.widgets.FloatSlider(name="bulk %", value=90, start=50, end=99, width=150, step=1)

@pn.depends(half_checkbox_cauchy.param.value, watch=True)
def invisible_L_cauchy(half):
    if half:
        L_input_cauchy.value = "0"
        L_input_cauchy.disabled = True
        # L_input_normal.visible = False
    else:
        L_input_cauchy.disabled = False
        # L_input_normal.visible=True

@pn.depends(L_input_cauchy.param.value, U_input_cauchy.param.value,
            bulk_slider_cauchy.param.value, half_checkbox_cauchy.param.value)
def cauchy_table(L, U, bulk, half):
    L, U, bulk = float(L), float(U), float(bulk)
    if half:
        leftover = 100 - bulk
        bulk = 100 - 2*leftover
        μ, σ = find_cauchy(-U, U, bulk/100, precision=10)
    else:
        μ, σ = find_cauchy(L, U, bulk/100, precision=10)

    return pn.pane.Markdown(f"""
        | param | value |
        | ----- | ----- |
        | μ | {np.round(μ, 4)} |
        | σ | {np.round(σ, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )
@pn.depends(L_input_cauchy.param.value, U_input_cauchy.param.value,
            bulk_slider_cauchy.param.value, half_checkbox_cauchy.param.value)
def dashboard_cauchy(L, U, bulk, half):
    L, U, bulk = float(L), float(U), float(bulk)
    if half:
        leftover = 100 - bulk
        bulk = 100 - 2*leftover
        μ, σ = find_cauchy(-U, U, bulk/100, precision=10)

        f_pdf = lambda arr, mu, sigma: scipy.stats.halfnorm.pdf(arr, mu, sigma)
        f_cdf = lambda arr, mu, sigma: scipy.stats.halfnorm.cdf(arr, mu, sigma)
        padding = U * 0.3
        x = np.linspace(0, U, 1_000)
        x_high = scipy.stats.norm.ppf([0.95], μ, σ)[0]
        x_full = np.linspace(0, max(U+padding, x_high), 1_000)
        pdf_full = f_pdf(x_full, μ, σ)
        cdf_full = f_cdf(x_full, μ, σ)

        x_patch = [0] + list(x) + [U]
        pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
        cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]

        row_pdfcdf = _pdfcdf_2p2b_plotter(0, U, μ, σ, f_pdf, f_cdf,
                x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_cauchy)
    else:
        μ, σ = find_cauchy(L, U, bulk/100, precision=10)

        f_pdf = lambda arr, mu, sigma: scipy.stats.cauchy.pdf(arr, mu, sigma)
        f_cdf = lambda arr, mu, sigma: scipy.stats.cauchy.cdf(arr, mu, sigma)

        padding = (U - L) * 0.3
        x = np.linspace(L, U, 1_000)
        x_low, x_high = scipy.stats.cauchy.ppf([0.05, 0.95], μ, σ)
        x_full = np.linspace(min(L-padding, x_low), max(U+padding, x_high), 1_000)
        pdf_full = f_pdf(x_full, μ, σ)
        cdf_full = f_cdf(x_full, μ, σ)

        x_patch = [L] + list(x) + [U]
        pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
        cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]
        row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, μ, σ, f_pdf, f_cdf,
            x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_cauchy)

    return row_pdfcdf

row_LUbulk_cauchy = pn.Row(L_input_cauchy, U_input_cauchy, bulk_slider_cauchy,
    pn.Column(pn.Spacer(height=10), half_checkbox_cauchy), pn.Spacer(width=10), cauchy_table)

layout_cauchy = pn.Column(row_LUbulk_cauchy, dashboard_cauchy, name="Cauchy")



# STUDENT-T
half_checkbox_studentt = pn.widgets.Checkbox(name='half', width=50, value=False)
ν_input_studentt = pn.widgets.TextInput(name="ν", value="3", width=55)
L_input_studentt = pn.widgets.TextInput(name="L", value="1", width=93)
U_input_studentt = pn.widgets.TextInput(name="U", value="10", width=93)
bulk_slider_studentt = pn.widgets.FloatSlider(name="bulk %", value=95, start=50, end=99, width=150, step=1, value_throttled=True)

@pn.depends(half_checkbox_studentt.param.value, watch=True)
def invisible_L_studentt(half):
    if half:
        L_input_studentt.value = "0"
        L_input_studentt.disabled = True
        # L_input_normal.visible = False
    else:
        L_input_studentt.disabled = False
        # L_input_normal.visible=True

@pn.depends(ν_input_studentt.param.value, L_input_studentt.param.value,
            U_input_studentt.param.value, bulk_slider_studentt.param.value,
            half_checkbox_studentt.param.value)
def studentt_table(ν, L, U, bulk, half):
    ν, L, U, bulk = float(ν), float(L), float(U), float(bulk)

    if half:
        leftover = 100 - bulk
        bulk = 100 - 2*leftover
        μ, σ = find_studentt(ν, -U, U, bulk/100, precision=10)
    else:
        μ, σ = find_studentt(ν, L, U, bulk/100, precision=10)

    return pn.pane.Markdown(f"""
        | param | value |
        | ----- | ----- |
        | μ | {np.round(μ, 4)} |
        | σ | {np.round(σ, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )
@pn.depends(ν_input_studentt.param.value, L_input_studentt.param.value,
            U_input_studentt.param.value, bulk_slider_studentt.param.value,
            half_checkbox_studentt.param.value)
def dashboard_studentt(ν, L, U, bulk, half):
    ν, L, U, bulk = float(ν), float(L), float(U), float(bulk)

    if half:
        leftover = 100 - bulk
        bulk = 100 - 2*leftover
        μ, σ = find_studentt(ν, -U, U, bulk/100, precision=10)

        f_pdf = lambda arr, mu, sigma: scipy.stats.t.pdf(arr, ν, mu, sigma)
        f_cdf = lambda arr, mu, sigma: scipy.stats.t.cdf(arr, ν, mu, sigma)
        padding = U * 0.3
        x = np.linspace(0, U, 1_000)
        x_high = scipy.stats.t.ppf([0.995], ν, μ, σ)[0]
        x_full = np.linspace(0, max(U+padding, x_high), 1_000)

        x_patch = [0] + list(x) + [U]

        pdf_full = f_pdf(x_full, μ, σ)
        cdf_full = f_cdf(x_full, μ, σ)

        pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
        cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]

        row_pdfcdf = _pdfcdf_2p2b_plotter(0, U, μ, σ, f_pdf, f_cdf,
                x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_studentt)
    else:
        μ, σ = find_studentt(ν, L, U, bulk/100, precision=10)

        f_pdf = lambda arr, mu, sigma: scipy.stats.t.pdf(arr, ν, mu, sigma)
        f_cdf = lambda arr, mu, sigma: scipy.stats.t.cdf(arr, ν, mu, sigma)

        padding = (U - L) * 0.3
        x = np.linspace(L, U, 1_000)
        x_low, x_high = scipy.stats.t.ppf([0.025, 0.975], ν, μ, σ)
        x_full = np.linspace(min(x_low, L-padding), max(x_high, U+padding), 1_000)

        pdf_full = f_pdf(x_full, μ, σ)
        cdf_full = f_cdf(x_full, μ, σ)

        x_patch = [L] + list(x) + [U]
        pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
        cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]
        row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, μ, σ, f_pdf, f_cdf,
            x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color_studentt)

    return row_pdfcdf

row_LUbulk_studentt = pn.Row(ν_input_studentt, L_input_studentt, U_input_studentt, bulk_slider_studentt,
    pn.Column(pn.Spacer(height=10), half_checkbox_studentt), pn.Spacer(width=10), studentt_table)
layout_studentt = pn.Column(row_LUbulk_studentt, dashboard_studentt, name="StudentT")





# GUMBEL

L_input_gumbel = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_gumbel = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_gumbel = pn.widgets.FloatSlider(name="bulk %", value=99, start=50, end=99, width=150, step=1)

@pn.depends(L_input_gumbel.param.value, U_input_gumbel.param.value, bulk_slider_gumbel.param.value)
def gumbel_table(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    μ, σ = find_gumbel(L, U, bulk=bulk/100, precision=10)
    return pn.pane.Markdown(f"""
        | param | value |
        | ----- | ----- |
        | μ | {np.round(μ, 4)} |
        | σ | {np.round(σ, 4)} |
        """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )

@pn.depends(L_input_gumbel.param.value, U_input_gumbel.param.value, bulk_slider_gumbel.param.value)
def dashboard_gumbel(L, U, bulk):
    L, U, bulk = float(L), float(U), float(bulk)
    μ, σ = find_gumbel(L, U, bulk=bulk/100, precision=4)

    f_pdf = lambda arr, mu, sigma: scipy.stats.gumbel_r.pdf(arr, loc=mu, scale=sigma)
    f_cdf = lambda arr, mu, sigma: scipy.stats.gumbel_r.cdf(arr, loc=mu, scale=sigma)

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.gumbel_r.ppf([0.005, 0.995], loc=μ, scale=σ)
    x_full = np.linspace(min(x_low, L-padding), max(x_high, U+padding), 1_000)

    pdf_full = f_pdf(x_full, μ, σ)
    cdf_full = f_cdf(x_full, μ, σ)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
    cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, μ, σ, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color_gumbel)

    return row_pdfcdf

row_LUbulk_gumbel = pn.Row(L_input_gumbel, U_input_gumbel,
                           bulk_slider_gumbel, pn.Spacer(width=80), gumbel_table)
layout_gumbel = pn.Column(row_LUbulk_gumbel, dashboard_gumbel, name="Gumbel")


# FULL DASHBOARD
md_title = pn.pane.Markdown("""Constructing Priors""",
                 style={"font-family":'GillSans', 'font-size':'24px'})

tabs = pn.Tabs(
    layout_normal, layout_studentt, layout_gumbel, layout_expon, layout_gamma, layout_invgamma,
    layout_weibull, layout_pareto, layout_lognormal, layout_cauchy)

prior_dashboard = pn.Column(md_title, tabs)
prior_dashboard.servable()
