import os
import sys

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import scipy.special

import bokeh.io

import panel as pn
pn.extension('katex')

import warnings
warnings.filterwarnings('ignore')

def style(p, autohide=False):
    p.title.text_font="Helvetica"
    p.title.text_font_size="16px"
    p.title.align="center"
    p.xaxis.axis_label_text_font="Helvetica"
    p.yaxis.axis_label_text_font="Helvetica"

    p.xaxis.axis_label_text_font_size="13px"
    p.yaxis.axis_label_text_font_size="13px"
    p.background_fill_alpha = 0
    if autohide: p.toolbar.autohide=True
    return p

from .prior_inverse_search import *
from . import blurb_normal
from . import blurb_studentt
from . import blurb_gumbel
from . import blurb_exponential
from . import blurb_gamma
from . import blurb_invgamma
from . import blurb_weibull
from . import blurb_pareto
from . import blurb_lognormal
from . import blurb_cauchy
from . import blurb_beta


# *********************************** COLORS ***********************************
purple = "#aa6abe"
lightpurple = "#BB8FCE"
blue = "#5499c7"
green = "#52be80"
pink = "#DD7295"
red = "#cd6155"
orange = "#E67E22"

color_normal = purple
color_studentt = blue
color_exponential = orange
color_gamma = green
color_invgamma = "orange"
color_weibull = "teal"
color_pareto = pink
color_lognormal = "#7162EE"
color_gumbel = "#C7545F"
color_cauchy = "#B5A642"
color_beta = "#000000"

color_null = "#cccccc"

# ********************** MISCELLANEOUS PLOTTING FUNCTIONS **********************
# 2p2b: 2 parameters (function call), 2 boundaries (2 circles)
def _pdfcdf_2p2b_plotter(
    L, U, mu, sigma, f_pdf, f_cdf,
    x_patch, pdf_patch, cdf_patch,
    x_full, pdf_full, cdf_full,
    color="purple", pad_left=False, pad_right=False
):
    x_range = (min(x_full), max(x_full))
    y_range = (-0.05, 1.05)
    if pad_left:
        x_range = (min(x_full) - 0.04*np.ptp(x_full), max(x_full))
        y_range = (-0.07, 1.05)
    if pad_right: # only used for the beta distribution
        x_range = (min(x_full)-0.02, max(x_full)+0.02)
        y_range = (-0.07, 1.05)

    tools = 'pan,box_zoom,wheel_zoom,reset'
    p_pdf = bokeh.plotting.figure(title="pdf", width=350, height=220, x_range=x_range, tools=tools)
    p_cdf = bokeh.plotting.figure(title="cdf", width=350, height=220, x_range=x_range, y_range=y_range, tools=tools)

    # shading pdf
    p_pdf.patch(x_patch, pdf_patch, color='#eaeaea')

    # horizontal lines cdf
    p_cdf.line((x_full[0], U), (cdf_patch[-2], cdf_patch[-2]), line_color="#bdbdbd", line_width=2.2, line_dash="dotdash")
    p_cdf.line((x_full[0], L), (cdf_patch[1], cdf_patch[1]), line_color="#bdbdbd", line_width=2.2, line_dash="dotdash")

    # curves
    p_pdf.line(x_full, pdf_full, line_width=3.5, color=color)
    p_cdf.line(x_full, cdf_full, line_width=3.5, color=color)

    # boundary pts
    p_pdf.circle(L, f_pdf(L, mu, sigma), size=9, line_width=2.5, line_color=color, fill_color="white")
    p_pdf.circle(U, f_pdf(U, mu, sigma), size=9, line_width=2.5, line_color=color, fill_color="white")

    # ************************  PDF HOVER BULK  ********************************
    # curves: hover bulk
    bulk_cdf = round(100*(f_cdf(U, mu, sigma) - f_cdf(L, mu, sigma)))
    cds_curves = bokeh.models.ColumnDataSource(data={"x_bulk": x_patch[1:-1], "pdf_bulk": pdf_patch[1:-1]})
    g_pdf = bokeh.models.Line(x="x_bulk", y="pdf_bulk", line_alpha=0.0)
    g_r_pdf = p_pdf.add_glyph(source_or_glyph=cds_curves, glyph=g_pdf)
    g_hover_pdf = bokeh.models.HoverTool(renderers=[g_r_pdf], mode='vline', line_policy="nearest",
        tooltips=f"""<div><div><div>
            <span style="font-size: 13px; font-family: Helvetica; color: {color}; font-weight: bold;"> bulk: </span>
            <span style="font-size: 13px; font-family: Helvetica; color: black"> {bulk_cdf}% </span>
        </div></div></div>""")
    p_pdf.add_tools(g_hover_pdf)

    # ***********************  PDF HOVER ENDPTS  *******************************
    cds_bounds = bokeh.models.ColumnDataSource(data={
        "L": [L], "U": [U], "pdf_low": [f_pdf(L, mu, sigma)], "pdf_high": [f_pdf(U, mu, sigma)]})
    gL_pdf = bokeh.models.Circle(x="L", y="pdf_low", size=9, line_width=2.5, line_color=color, fill_color="white", name="bounds")
    gU_pdf = bokeh.models.Circle(x="U", y="pdf_high", size=9, line_width=2.5, line_color=color, fill_color="white", name="bounds")
    gL_r = p_pdf.add_glyph(source_or_glyph=cds_bounds, glyph=gL_pdf)
    gU_r = p_pdf.add_glyph(source_or_glyph=cds_bounds, glyph=gU_pdf)

    tooltips_L_pdf = f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> &nbsp;&nbsp; L: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{L} </span> </div></div></div>""" + \
        f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> pdf: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{pdf_low}{0.000} </span> </div></div></div>"""

    tooltips_U_pdf = f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> &nbsp;&nbsp; U: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{U} </span> </div></div></div>""" + \
        f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> pdf: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{pdf_high}{0.000} </span> </div></div></div>"""
    gL_hover = bokeh.models.HoverTool(renderers=[gL_r], mode='vline', line_policy="nearest",
        tooltips=tooltips_L_pdf)
    gU_hover = bokeh.models.HoverTool(renderers=[gU_r], mode='vline', line_policy="nearest",
        tooltips=tooltips_U_pdf)
    p_pdf.add_tools(gL_hover)
    p_pdf.add_tools(gU_hover)

    # ************************  CDF HOVER BULK  ********************************
    # curves: hover bulk
    cds_curves = bokeh.models.ColumnDataSource(data={"x_full": x_full, "cdf_full": cdf_full})
    g_cdf = bokeh.models.Line(x="x_full", y="cdf_full", line_alpha=0.0)
    g_r_cdf = p_cdf.add_glyph(source_or_glyph=cds_curves, glyph=g_cdf)
    g_hover_cdf = bokeh.models.HoverTool(renderers=[g_r_cdf], mode='vline', line_policy="nearest",
        tooltips=f"""<div><div><div>
            <span style="font-size: 13px; font-family: Helvetica; color: {color}; font-weight: bold;"> &nbsp;&nbsp; y: </span>
            <span style="font-size: 13px; font-family: Helvetica; color: black"> @x_full </span> <br>
            <span style="font-size: 13px; font-family: Helvetica; color: {color}; font-weight: bold;"> cdf: </span>
            <span style="font-size: 13px; font-family: Helvetica; color: black"> @cdf_full </span>
        </div></div></div>""")
    p_cdf.add_tools(g_hover_cdf)

    # ***********************  CDF HOVER ENDPTS  *******************************
    cds_bounds = bokeh.models.ColumnDataSource(data={
        "L": [L], "U": [U], "cdf_low": [f_cdf(L, mu, sigma)], "cdf_high": [f_cdf(U, mu, sigma)]})
    gL_cdf = bokeh.models.Circle(x="L", y="cdf_low", size=9, line_width=2.5, line_color=color, fill_color="white", name="bounds")
    gU_cdf = bokeh.models.Circle(x="U", y="cdf_high", size=9, line_width=2.5, line_color=color, fill_color="white", name="bounds")
    gL_r = p_cdf.add_glyph(source_or_glyph=cds_bounds, glyph=gL_cdf)
    gU_r = p_cdf.add_glyph(source_or_glyph=cds_bounds, glyph=gU_cdf)

    tooltips_L_cdf = f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> &nbsp;&nbsp; L: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{L} </span> </div></div></div>""" + \
        f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> cdf: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{cdf_low}{0.000} </span> </div></div></div>"""

    tooltips_U_cdf = f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> &nbsp;&nbsp; U: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{U} </span> </div></div></div>""" + \
        f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> cdf: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{cdf_high}{0.000} </span> </div></div></div>"""
    gL_hover = bokeh.models.HoverTool(renderers=[gL_r], mode='vline', line_policy="nearest",
        tooltips=tooltips_L_cdf)
    gU_hover = bokeh.models.HoverTool(renderers=[gU_r], mode='vline', line_policy="nearest",
        tooltips=tooltips_U_cdf)
    p_cdf.add_tools(gL_hover)
    p_cdf.add_tools(gU_hover)

    # **************************************************************************
    row_pdfcdf = pn.Row(style(p_pdf, autohide=True), style(p_cdf, autohide=True))

    return row_pdfcdf

# 1p1b: 1 parameter (function call), 1 boundary (1 circle), expon and pareto
def _pdfcdf_1p1b_plotter(
    U, beta, f_pdf, f_cdf,
    x_patch, pdf_patch, cdf_patch,
    x_full, pdf_full, cdf_full,
    color="green"
):
    x_range = (min(x_full), max(x_full))
    tools = 'pan,box_zoom,wheel_zoom,reset'
    p_pdf = bokeh.plotting.figure(title="pdf", width=350, height=220, x_range=x_range, tools=tools)
    p_cdf = bokeh.plotting.figure(title="cdf", width=350, height=220, x_range=x_range, y_range=(-0.05, 1.05), tools=tools)

    # shading pdf
    p_pdf.patch(x_patch, pdf_patch, color='#eaeaea')

    # horizontal lines for cdf
    p_cdf.line((x_full[0], U), (cdf_patch[-2], cdf_patch[-2]), line_color="#bdbdbd", line_width=2.2, line_dash="dotdash")

    # curves
    p_pdf.line(x_full, pdf_full, line_width=3.5, color=color)
    p_cdf.line(x_full, cdf_full, line_width=3.5, color=color)

    # boundaries
    p_pdf.circle(U, f_pdf(U, beta), size=9, line_width=2.5, line_color=color, fill_color="white")
    p_cdf.circle(U, f_cdf(U, beta), size=9, line_width=2.5, line_color=color, fill_color="white")

    # ************************  PDF HOVER BULK  ********************************
    # curves: hover bulk
    bulk_cdf = round(100*(f_cdf(U, beta)))
    cds_curves = bokeh.models.ColumnDataSource(data={"x_bulk": x_patch[1:-1], "pdf_bulk": pdf_patch[1:-1]})
    g_pdf = bokeh.models.Line(x="x_bulk", y="pdf_bulk", line_alpha=0.0)
    g_r_pdf = p_pdf.add_glyph(source_or_glyph=cds_curves, glyph=g_pdf)
    g_hover_pdf = bokeh.models.HoverTool(renderers=[g_r_pdf], mode='vline', line_policy="nearest",
        tooltips=f"""<div><div><div>
            <span style="font-size: 13px; font-family: Helvetica; color: {color}; font-weight: bold;"> bulk: </span>
            <span style="font-size: 13px; font-family: Helvetica; color: black"> {bulk_cdf}% </span>
        </div></div></div>""")
    p_pdf.add_tools(g_hover_pdf)

    # ************************  CDF HOVER BULK  ********************************
    # curves: hover bulk
    cds_curves = bokeh.models.ColumnDataSource(data={"x_full": x_full, "cdf_full": cdf_full})
    g_cdf = bokeh.models.Line(x="x_full", y="cdf_full", line_alpha=0.0)
    g_r_cdf = p_cdf.add_glyph(source_or_glyph=cds_curves, glyph=g_cdf)
    g_hover_cdf = bokeh.models.HoverTool(renderers=[g_r_cdf], mode='vline', line_policy="nearest",
        tooltips=f"""<div><div><div>
            <span style="font-size: 13px; font-family: Helvetica; color: {color}; font-weight: bold;"> &nbsp;&nbsp; y: </span>
            <span style="font-size: 13px; font-family: Helvetica; color: black"> @x_full </span> <br>
            <span style="font-size: 13px; font-family: Helvetica; color: {color}; font-weight: bold;"> cdf: </span>
            <span style="font-size: 13px; font-family: Helvetica; color: black"> @cdf_full </span>
        </div></div></div>""")
    p_cdf.add_tools(g_hover_cdf)

    # ***********************  PDF HOVER ENDPTS  *******************************
    cds_bounds = bokeh.models.ColumnDataSource(data={"U": [U], "pdf_high": [f_pdf(U, beta)]})
    gU_pdf = bokeh.models.Circle(x="U", y="pdf_high", size=9, line_width=2.5, line_color=color, fill_color="white")
    gU_r = p_pdf.add_glyph(source_or_glyph=cds_bounds, glyph=gU_pdf)

    tooltips_U_pdf = f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> &nbsp;&nbsp; U: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{U} </span> </div></div></div>""" + \
        f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> pdf: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{pdf_high}{0.00} </span> </div></div></div>"""
    gU_hover = bokeh.models.HoverTool(renderers=[gU_r], mode='vline', line_policy="nearest",
        tooltips=tooltips_U_pdf)
    p_pdf.add_tools(gU_hover)

    # ***********************  CDF HOVER ENDPTS  *******************************
    cds_bounds = bokeh.models.ColumnDataSource(data={"U": [U], "cdf_high": [f_cdf(U, beta)]})
    gU_cdf = bokeh.models.Circle(x="U", y="cdf_high", size=9, line_width=2.5, line_color=color, fill_color="white")
    gU_r = p_cdf.add_glyph(source_or_glyph=cds_bounds, glyph=gU_cdf)

    tooltips_U_cdf = f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> &nbsp;&nbsp; U: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{U} </span> </div></div></div>""" + \
        f"""<div><div><div> <span style="font-size: 13px;
        font-family: Helvetica; color: {color}; font-weight: bold;"> cdf: </span>""" + \
        """<span style="font-size: 13px; font-family: Helvetica; color: black">
            @{cdf_high}{0.00} </span> </div></div></div>"""
    gU_hover = bokeh.models.HoverTool(renderers=[gU_r], mode='vline', line_policy="nearest",
        tooltips=tooltips_U_cdf)
    p_cdf.add_tools(gU_hover)

    # **************************************************************************
    row_pdfcdf = pn.Row(style(p_pdf, autohide=True), style(p_cdf, autohide=True))

    return row_pdfcdf


# ********************************** WIDGETS ***********************************
half_checkbox_normal = pn.widgets.Checkbox(name='half', width=50, value=False)
L_input_normal = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_normal = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_normal = pn.widgets.FloatSlider(name="bulk %", value=99, start=50, end=99, width=150, step=1, value_throttled=True)

L_input_lognormal = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_lognormal = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_lognormal = pn.widgets.FloatSlider(name="bulk %", value=90, start=50, end=99, width=150, step=1, value_throttled=True)

L_input_gamma = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_gamma = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_gamma = pn.widgets.FloatSlider(name="bulk %", value=99, start=50, end=99, width=150, step=1, value_throttled=True)

L_input_invgamma = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_invgamma = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_invgamma = pn.widgets.FloatSlider(name="bulk %", value=95, start=50, end=99, width=150, step=1, value_throttled=True)

L_input_weibull = pn.widgets.TextInput(name="L", value="0.1", width=130)
U_input_weibull = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_weibull = pn.widgets.FloatSlider(name="bulk %", value=99, start=50, end=99, width=150, step=1, value_throttled=True)

U_input_expon = pn.widgets.TextInput(name="U", value="10", width=130)
Uppf_slider_expon = pn.widgets.FloatSlider(name="Uppf %", value=99, start=50, end=99, width=150, step=1, value_throttled=True)

ymin_input_pareto = pn.widgets.TextInput(name="ymin", value="0.1", width=70)
U_input_pareto = pn.widgets.TextInput(name="U", value="1", width=130)
Uppf_slider_pareto = pn.widgets.FloatSlider(name="Uppf %", value=99, start=50, end=99, width=150, step=1, value_throttled=True)

half_checkbox_cauchy = pn.widgets.Checkbox(name='half', width=50, value=False)
L_input_cauchy = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_cauchy = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_cauchy = pn.widgets.FloatSlider(name="bulk %", value=90, start=50, end=99, width=150, step=1, value_throttled=True)

half_checkbox_studentt = pn.widgets.Checkbox(name='half', width=50, value=False)
ν_input_studentt = pn.widgets.TextInput(name="ν", value="3", width=55)
L_input_studentt = pn.widgets.TextInput(name="L", value="1", width=93)
U_input_studentt = pn.widgets.TextInput(name="U", value="10", width=93)
bulk_slider_studentt = pn.widgets.FloatSlider(name="bulk %", value=95, start=50, end=99, width=150, step=1, value_throttled=True)

L_input_gumbel = pn.widgets.TextInput(name="L", value="1", width=130)
U_input_gumbel = pn.widgets.TextInput(name="U", value="10", width=130)
bulk_slider_gumbel = pn.widgets.FloatSlider(name="bulk %", value=99, start=50, end=99, width=150, step=1, value_throttled=True)

L_input_beta = pn.widgets.TextInput(name="L", value="0.1", width=130)
U_input_beta = pn.widgets.TextInput(name="U", value="0.9", width=130)
toggle_beta = pn.widgets.RadioButtonGroup(options=["center", "asymm"], width=150, height=30) # get inline F
bulk_slider_beta = pn.widgets.FloatSlider(name="bulk %", value=75, start=50, end=99,
    step=1, value_throttled=True, width=150, margin=(6, 10, 0, 10))
range_slider_beta = pn.widgets.RangeSlider(name="bulk %", start=1, end=99, step=1,
    value=(5, 85), width=150, visible=False, value_throttled=True, margin=(6, 10, 0, 10))

@pn.depends(toggle_beta.param.value, watch=True)
def invisible_bulkrangeslider_beta(toggle):
    if toggle == "center":
        range_slider_beta.visible, bulk_slider_beta.visible = False, True
    elif toggle == "asymm":
        bulk_slider_beta.visible, range_slider_beta.visible = False, True

# ********************************** TABLES ************************************
@pn.depends(L_input_normal.param.value, U_input_normal.param.value,
            bulk_slider_normal.param.value, half_checkbox_normal.param.value)
def normal_table(L, U, bulk, half):
    try:
        L, U, bulk = float(L), float(U), float(bulk)

        if half:
            μ, σ = find_normal(-U, U, bulk/100, precision=10)
        else:
            μ, σ = find_normal(L, U, bulk/100, precision=10)

        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ | {np.round(μ, 4)} |
            | σ | {np.round(σ, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ     |       |
            | σ     |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

@pn.depends(L_input_lognormal.param.value, U_input_lognormal.param.value, bulk_slider_lognormal.param.value)
def lognormal_table(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        μ, σ = find_lognormal(L, U, bulk/100, precision=10)

        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ | {np.round(μ, 4)} |
            | σ | {np.round(σ, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            |  μ    |       |
            |  σ    |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

@pn.depends(L_input_gamma.param.value, U_input_gamma.param.value, bulk_slider_gamma.param.value)
def gamma_table(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        α, β = find_gamma(L, U, bulk=bulk/100, precision=10)
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | α | {np.round(α, 4)} |
            | β | {np.round(β, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | α     |       |
            | β     |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

@pn.depends(L_input_invgamma.param.value, U_input_invgamma.param.value, bulk_slider_invgamma.param.value)
def invgamma_table(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        α, β = find_invgamma(L, U, bulk=bulk/100, precision=10)

        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | α | {np.round(α, 4)} |
            | β | {np.round(β, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | α     |       |
            | β     |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

@pn.depends(L_input_weibull.param.value, U_input_weibull.param.value, bulk_slider_weibull.param.value)
def weibull_table(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        α, σ = find_weibull(L, U, bulk=bulk/100, precision=10)

        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | α | {np.round(α, 4)} |
            | σ | {np.round(σ, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | α     |       |
            | σ     |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

@pn.depends(U_input_expon.param.value, Uppf_slider_expon.param.value)
def expon_table(U, Uppf):
    try:
        U, Uppf = float(U), float(Uppf)
        β = find_exponential(U, Uppf/100, precision=10)
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----------: | ----------- |
            | β | {np.round(β, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----------: | ----------- |
            | β |    |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

@pn.depends(ymin_input_pareto.param.value, U_input_pareto.param.value, Uppf_slider_pareto.param.value)
def pareto_table(ymin, U, Uppf):
    try:
        ymin, U, Uppf = float(ymin), float(U), float(Uppf)
        α = find_pareto(ymin, U, Uppf/100, precision=10)
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----------: | ----------- |
            | α | {np.round(α, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----------: | ----------- |
            | α           |             |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
@pn.depends(L_input_cauchy.param.value, U_input_cauchy.param.value,
            bulk_slider_cauchy.param.value, half_checkbox_cauchy.param.value)
def cauchy_table(L, U, bulk, half):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        if half:
            μ, σ = find_cauchy(-U, U, bulk/100, precision=10)
        else:
            μ, σ = find_cauchy(L, U, bulk/100, precision=10)

        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ | {np.round(μ, 4)} |
            | σ | {np.round(σ, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ     |       |
            | σ     |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

@pn.depends(ν_input_studentt.param.value, L_input_studentt.param.value,
            U_input_studentt.param.value, bulk_slider_studentt.param.value,
            half_checkbox_studentt.param.value)
def studentt_table(ν, L, U, bulk, half):
    try:
        ν, L, U, bulk = float(ν), float(L), float(U), float(bulk)
        if half:
            μ, σ = find_studentt(ν, -U, U, bulk/100, precision=10)
        else:
            μ, σ = find_studentt(ν, L, U, bulk/100, precision=10)

        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ | {np.round(μ, 4)} |
            | σ | {np.round(σ, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ     |       |
            | σ     |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

@pn.depends(L_input_gumbel.param.value, U_input_gumbel.param.value, bulk_slider_gumbel.param.value)
def gumbel_table(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        μ, σ = find_gumbel(L, U, bulk=bulk/100, precision=10)
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ | {np.round(μ, 4)} |
            | σ | {np.round(σ, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | μ     |       |
            | σ     |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
    )

@pn.depends(L_input_beta.param.value, U_input_beta.param.value, bulk_slider_beta.param.value)
def beta_table(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        α, β = find_beta(L, U, bulk=bulk/100, precision=10)
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | α | {np.round(α, 4)} |
            | β | {np.round(β, 4)} |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )
    except:
        return pn.pane.Markdown(f"""
            | param | value |
            | :-----: | ----- |
            | α     |       |
            | β     |       |
            """, style={'border':'4px solid lightgrey', 'border-radius':'5px'}
        )

# *********************************** NORMAL ***********************************
@pn.depends(half_checkbox_normal.param.value, watch=True)
def invisible_L_normal(half):
    if half:
        L_input_normal.value = "0"
        L_input_normal.disabled = True
    else:
        L_input_normal.disabled = False

@pn.depends(L_input_normal.param.value, U_input_normal.param.value,
            bulk_slider_normal.param.value, half_checkbox_normal.param.value)
def dashboard_normal(L, U, bulk, half):

    if half:
        try:
            L, U, bulk = float(L), float(U), float(bulk)
            μ, σ, _, U = find_normal(-U, U, bulk/100, precision=10, return_bounds=True)
            color = color_normal
        except:
            μ, σ = 1, 1
            color = color_null

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
                x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color, pad_left=True)

    else:
        try:
            L, U, bulk = float(L), float(U), float(bulk)
            μ, σ, L, U = find_normal(L, U, bulk/100, precision=10, return_bounds=True)
            color = color_normal
        except:
            μ, σ = 1, 1
            color = color_null

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
                x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color)

    return row_pdfcdf

# ********************************* LOG-NORMAL *********************************
@pn.depends(L_input_lognormal.param.value, U_input_lognormal.param.value, bulk_slider_lognormal.param.value)
def dashboard_lognormal(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        μ, σ, L, U = find_lognormal(L, U, bulk/100, precision=10, return_bounds=True)
        color = color_lognormal
    except:
        μ, σ = 1, 1
        color = color_null

    f_pdf = lambda arr, mu, sigma: scipy.stats.lognorm.pdf(arr, sigma, loc=0, scale=np.exp(mu))
    f_cdf = lambda arr, mu, sigma: scipy.stats.lognorm.cdf(arr, sigma, loc=0, scale=np.exp(mu))

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.lognorm.ppf([0.05, 0.95], σ, loc=0, scale=np.exp(μ))
    x_full = np.linspace(max(0, L-padding), max(x_high, U+padding), 1_000)
    x_full = np.sort(np.append(x_full, L))

    pdf_full = f_pdf(x_full, μ, σ)
    cdf_full = f_cdf(x_full, μ, σ)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
    cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, μ, σ, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color, pad_left=True)

    return row_pdfcdf

# *********************************** GAMMA ***********************************
@pn.depends(L_input_gamma.param.value, U_input_gamma.param.value, bulk_slider_gamma.param.value)
def dashboard_gamma(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        α, β, L, U = find_gamma(L, U, bulk=bulk/100, precision=10, return_bounds=True)
        color = color_gamma
    except:
        α, β = 1, 1
        color = color_null

    f_pdf = lambda arr, alpha, beta: scipy.stats.gamma.pdf(arr, alpha, scale=1/beta)
    f_cdf = lambda arr, alpha, beta: scipy.stats.gamma.cdf(arr, alpha, scale=1/beta)

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.gamma.ppf([0.025, 0.975], α, scale=1/β)
    x_full = np.linspace(max(0, L-padding), max(x_high, U+padding), 1_000)
    x_full = np.sort(np.append(x_full, L))

    pdf_full = f_pdf(x_full, α, β)
    cdf_full = f_cdf(x_full, α, β)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, α, β)) + [0]
    cdf_patch = [0] + list(f_cdf(x, α, β)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, α, β, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color, pad_left=True)

    return row_pdfcdf

# ******************************* INVERSE GAMMA *******************************
@pn.depends(L_input_invgamma.param.value, U_input_invgamma.param.value, bulk_slider_invgamma.param.value)
def dashboard_invgamma(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        α, β, L, U = find_invgamma(L, U, bulk=bulk/100, precision=10, return_bounds=True)
        color = color_invgamma
    except:
        α, β = 1, 1
        color = color_null

    f_pdf = lambda arr, alpha, beta: scipy.stats.invgamma.pdf(arr, alpha, scale=beta)
    f_cdf = lambda arr, alpha, beta: scipy.stats.invgamma.cdf(arr, alpha, scale=beta)

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.invgamma.ppf([0.05, 0.95], α, scale=β)
    x_full = np.linspace(max(0, L-padding), max(x_high, U+padding), 1_000)
    x_full = np.sort(np.append(x_full, L))

    pdf_full = f_pdf(x_full, α, β)
    cdf_full = f_cdf(x_full, α, β)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, α, β)) + [0]
    cdf_patch = [0] + list(f_cdf(x, α, β)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, α, β, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color, pad_left=True)

    return row_pdfcdf

# ********************************** WEIBULL **********************************
@pn.depends(L_input_weibull.param.value, U_input_weibull.param.value, bulk_slider_weibull.param.value)
def dashboard_weibull(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        α, σ, L, U = find_weibull(L, U, bulk=bulk/100, precision=10, return_bounds=True)
        color = color_weibull
    except:
        α, σ = 1, 1
        color = color_null

    f_pdf = lambda arr, alpha, sigma: scipy.stats.weibull_min.pdf(arr, alpha, scale=sigma)
    f_cdf = lambda arr, alpha, sigma: scipy.stats.weibull_min.cdf(arr, alpha, scale=sigma)

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.weibull_min.ppf([0.05, 0.95], α, scale=σ)
    x_full = np.linspace(max(0, L-padding), max(x_high, U+padding), 1_000)
    x_full = np.sort(np.append(x_full, L))

    pdf_full = f_pdf(x_full, α, σ)
    cdf_full = f_cdf(x_full, α, σ)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, α, σ)) + [0]
    cdf_patch = [0] + list(f_cdf(x, α, σ)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, α, σ, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color, pad_left=True)

    return row_pdfcdf

# ********************************* EXPONENTIAL *******************************
@pn.depends(U_input_expon.param.value, Uppf_slider_expon.param.value)
def dashboard_exponential(U, Uppf):
    try:
        U, Uppf = float(U), float(Uppf)
        β, U = find_exponential(U, Uppf/100, precision=10, return_bounds=True)
        color = color_exponential
    except:
        β = 1
        color = color_null
    f_pdf = lambda arr, beta: scipy.stats.expon.pdf(arr, loc=0, scale=1/beta)
    f_cdf = lambda arr, beta: scipy.stats.expon.cdf(arr, loc=0, scale=1/beta)

    padding = U * 0.05
    x = np.linspace(0, U, 1_000)
    x_low, x_high = scipy.stats.expon.ppf([0.005, 0.995], loc=0, scale=1/β)
    x_full = np.linspace(0, x_high, 1_000)

    pdf_full = f_pdf(x_full, β)
    cdf_full = f_cdf(x_full, β)

    x_patch = [0] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, β)) + [0]
    cdf_patch = [0] + list(f_cdf(x, β)) + [0]
    row_pdfcdf = _pdfcdf_1p1b_plotter(U, β, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color)

    return row_pdfcdf

# *********************************** PARETO ***********************************
@pn.depends(ymin_input_pareto.param.value, U_input_pareto.param.value, Uppf_slider_pareto.param.value)
def dashboard_pareto(ymin, U, Uppf):
    try:
        ymin, U, Uppf = float(ymin), float(U), float(Uppf)
        α, U = find_pareto(ymin, U, Uppf/100, precision=10, return_bounds=True)
        color = color_pareto
    except:
        α = 1
        color = color_null

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
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color)

    return row_pdfcdf

# *********************************** CAUCHY ***********************************
@pn.depends(half_checkbox_cauchy.param.value, watch=True)
def invisible_L_cauchy(half):
    if half:
        L_input_cauchy.value = "0"
        L_input_cauchy.disabled = True
    else:
        L_input_cauchy.disabled = False

@pn.depends(L_input_cauchy.param.value, U_input_cauchy.param.value,
            bulk_slider_cauchy.param.value, half_checkbox_cauchy.param.value)
def dashboard_cauchy(L, U, bulk, half):
    if half:
        try:
            if U < 0:
                color = color_null
                U = np.abs(U)
            else:
                L, U, bulk = float(L), float(U), float(bulk)
                μ, σ, _, U = find_cauchy(-U, U, bulk/100, precision=10, return_bounds=True)
                color = color_cauchy
        except:
            μ, σ = 1, 1
            color = color_null
        f_pdf = lambda arr, mu, sigma: scipy.stats.halfcauchy.pdf(arr, mu, sigma)
        f_cdf = lambda arr, mu, sigma: scipy.stats.halfcauchy.cdf(arr, mu, sigma)
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
                x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color, pad_left=True)
    else:
        try:
            L, U, bulk = float(L), float(U), float(bulk)
            μ, σ, L, U = find_cauchy(L, U, bulk/100, precision=10, return_bounds=True)
            color = color_cauchy
        except:
            μ, σ = 1, 1
            color = color_null

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
            x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color)

    return row_pdfcdf

# ********************************* STUDENT-T *********************************
@pn.depends(half_checkbox_studentt.param.value, watch=True)
def invisible_L_studentt(half):
    if half:
        L_input_studentt.value = "0"
        L_input_studentt.disabled = True
    else:
        L_input_studentt.disabled = False

@pn.depends(ν_input_studentt.param.value, L_input_studentt.param.value,
            U_input_studentt.param.value, bulk_slider_studentt.param.value,
            half_checkbox_studentt.param.value)
def dashboard_studentt(ν, L, U, bulk, half):
    ν, L, U, bulk = float(ν), float(L), float(U), float(bulk)

    if half:
        try:
            μ, σ, _, U = find_studentt(ν, -U, U, bulk/100, precision=10, return_bounds=True)
            color = color_studentt
        except:
            ν, μ, σ = 10, (U-L)/2+L, 1
            color = color_null

        # half student-t not implemented in scipy
        f_pdf = lambda arr, mu, sigma: 2*scipy.stats.t.pdf(arr, ν, mu, sigma)
        f_cdf = lambda arr, mu, sigma: 2*scipy.stats.t.cdf(arr, ν, mu, sigma)-1
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
                x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color, pad_left=True)
    else:
        try:
            ν, L, U, bulk = float(ν), float(L), float(U), float(bulk)
            μ, σ, L, U = find_studentt(ν, L, U, bulk/100, precision=10, return_bounds=True)
            color = color_studentt
        except:
            ν, μ, σ = 10, (U-L)/2+L, 1
            color = color_null

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
            x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color)

    return row_pdfcdf

# *********************************** GUMBEL ***********************************
@pn.depends(L_input_gumbel.param.value, U_input_gumbel.param.value, bulk_slider_gumbel.param.value)
def dashboard_gumbel(L, U, bulk):
    try:
        L, U, bulk = float(L), float(U), float(bulk)
        μ, σ, L, U = find_gumbel(L, U, bulk=bulk/100, precision=10, return_bounds=True)
        color = color_gumbel
    except:
        μ, σ = 1, 1
        color = color_null

    f_pdf = lambda arr, mu, sigma: scipy.stats.gumbel_r.pdf(arr, loc=mu, scale=sigma)
    f_cdf = lambda arr, mu, sigma: scipy.stats.gumbel_r.cdf(arr, loc=mu, scale=sigma)

    padding = (U - L) * 0.3
    x = np.linspace(L, U, 1_000)
    x_low, x_high = scipy.stats.gumbel_r.ppf([0.005, 0.995], loc=μ, scale=σ)
    x_full = np.linspace(min(x_low, L-padding), max(x_high, U+padding), 1_000)
    x_full = np.sort(np.append(x_full, L))

    pdf_full = f_pdf(x_full, μ, σ)
    cdf_full = f_cdf(x_full, μ, σ)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, μ, σ)) + [0]
    cdf_patch = [0] + list(f_cdf(x, μ, σ)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, μ, σ, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color)

    return row_pdfcdf

# *********************************** BETA ***********************************
@pn.depends(L_input_beta.param.value, U_input_beta.param.value, toggle_beta.param.value,
    bulk_slider_beta.param.value, range_slider_beta.param.value)
def dashboard_beta(L, U, toggle, bulk, rnge):
    try:
        L, U = float(L), float(U)
        if toggle == "center":
            α, β, L, U = find_beta(L, U, bulk=bulk/100, precision=10, return_bounds=True)

        if toggle == "asymm":
            Lppf, Uppf = rnge[0], rnge[1]
            α, β, L, U = find_beta(L, U, Lppf=Lppf/100, Uppf=Uppf/100, bulk=None,
                precision=10, return_bounds=True)
        color = color_beta
    except:
        α, β = 1, 1
        color = color_null

    f_pdf = lambda arr, alpha, beta: scipy.stats.beta.pdf(arr, alpha, beta)
    f_cdf = lambda arr, alpha, beta: scipy.stats.beta.cdf(arr, alpha, beta)

    x = np.linspace(L, U, 1_000)
    x_low, x_high = 0, 1
    x_full = np.linspace(0, 1, 1_000)
    x_full = np.sort(np.append(x_full, L))

    pdf_full = f_pdf(x_full, α, β)
    cdf_full = f_cdf(x_full, α, β)

    x_patch = [L] + list(x) + [U]
    pdf_patch = [0] + list(f_pdf(x, α, β)) + [0]
    cdf_patch = [0] + list(f_cdf(x, α, β)) + [0]
    row_pdfcdf = _pdfcdf_2p2b_plotter(L, U, α, β, f_pdf, f_cdf,
        x_patch, pdf_patch, cdf_patch, x_full, pdf_full, cdf_full, color=color, pad_right=True)

    return row_pdfcdf

#  ***************************************************************************************
#  *********************************** FULL DASHBOARD ************************************
#  ***************************************************************************************
row_LUbulk_beta = pn.Row(
    pn.Spacer(width=5),
    pn.Column(pn.Spacer(height=0), L_input_beta),
    pn.Column(pn.Spacer(height=0), U_input_beta),
    pn.Spacer(width=5),
    pn.Column(pn.Row(bulk_slider_beta, range_slider_beta), toggle_beta),
    pn.Spacer(width=50), beta_table)

row_LUbulk_normal = pn.Row(L_input_normal, U_input_normal, bulk_slider_normal,
    pn.Column(pn.Spacer(height=10), half_checkbox_normal), pn.Spacer(width=11), normal_table)
row_LUbulk_lognormal = pn.Row(L_input_lognormal, U_input_lognormal,
    bulk_slider_lognormal, pn.Spacer(width=80), lognormal_table)
row_LUbulk_gamma = pn.Row(L_input_gamma, U_input_gamma, bulk_slider_gamma,
    pn.Spacer(width=80), gamma_table)
row_LUbulk_invgamma = pn.Row(L_input_invgamma, U_input_invgamma,
    bulk_slider_invgamma, pn.Spacer(width=80), invgamma_table)
row_LUbulk_weibull = pn.Row(L_input_weibull, U_input_weibull,
    bulk_slider_weibull, pn.Spacer(width=80), weibull_table)
row_UUppf_expon = pn.Row(U_input_expon, Uppf_slider_expon,
    pn.Spacer(width=230), expon_table)
row_UUppf_pareto = pn.Row(ymin_input_pareto, U_input_pareto,
    Uppf_slider_pareto, pn.Spacer(width=140), pareto_table)
row_LUbulk_cauchy = pn.Row(L_input_cauchy, U_input_cauchy, bulk_slider_cauchy,
    pn.Column(pn.Spacer(height=10), half_checkbox_cauchy), pn.Spacer(width=10), cauchy_table)
row_LUbulk_studentt = pn.Row(ν_input_studentt, L_input_studentt, U_input_studentt, bulk_slider_studentt,
    pn.Column(pn.Spacer(height=10), half_checkbox_studentt), pn.Spacer(width=10), studentt_table)
row_LUbulk_gumbel = pn.Row(L_input_gumbel, U_input_gumbel, bulk_slider_gumbel,
    pn.Spacer(width=80), gumbel_table)


def bayesian_priors(description=False):
    md_title = pn.pane.Markdown("""Constructing Priors""",
        style={"font-family":'GillSans', 'font-size':'24px'})

    def wrap(x, name):
        return pn.Row(pn.Spacer(width=1), pn.Column(*x), name=name)


    if description:
        layout_normal = wrap([row_LUbulk_normal, dashboard_normal, blurb_normal.desc], "Normal")
        layout_lognormal = wrap([row_LUbulk_lognormal, dashboard_lognormal, blurb_lognormal.desc], "LogNormal")
        layout_gamma = wrap([row_LUbulk_gamma, dashboard_gamma, blurb_gamma.desc], "Gamma")
        layout_invgamma = wrap([row_LUbulk_invgamma, dashboard_invgamma, blurb_invgamma.desc], "InvGamma")
        layout_weibull = wrap([row_LUbulk_weibull, dashboard_weibull, blurb_weibull.desc], "Weibull")
        layout_expon = wrap([row_UUppf_expon, pn.Spacer(height=12),
            dashboard_exponential, blurb_exponential.desc], "Expon")
        layout_pareto = wrap([row_UUppf_pareto, pn.Spacer(height=12), dashboard_pareto, blurb_pareto.desc], "Pareto")
        layout_cauchy = wrap([row_LUbulk_cauchy, dashboard_cauchy, blurb_cauchy.desc], "Cauchy")
        layout_studentt = wrap([row_LUbulk_studentt, dashboard_studentt, blurb_studentt.desc], "StudentT")
        layout_gumbel = wrap([row_LUbulk_gumbel, dashboard_gumbel, blurb_gumbel.desc], "Gumbel")
        layout_beta = wrap([row_LUbulk_beta, dashboard_beta, blurb_beta.desc], "Beta")

    else:
        layout_normal = wrap([row_LUbulk_normal, dashboard_normal], "Normal")
        layout_lognormal = wrap([row_LUbulk_lognormal, dashboard_lognormal], "LogNormal")
        layout_gamma = wrap([row_LUbulk_gamma, dashboard_gamma], "Gamma")
        layout_invgamma = wrap([row_LUbulk_invgamma, dashboard_invgamma], "InvGamma")
        layout_weibull = wrap([row_LUbulk_weibull, dashboard_weibull], "Weibull")
        layout_expon = wrap([row_UUppf_expon, pn.Spacer(height=12), dashboard_exponential], "Expon")
        layout_pareto = wrap([row_UUppf_pareto, pn.Spacer(height=12), dashboard_pareto], "Pareto")
        layout_cauchy = wrap([row_LUbulk_cauchy, dashboard_cauchy], "Cauchy")
        layout_studentt = wrap([row_LUbulk_studentt, dashboard_studentt], "StudentT")
        layout_gumbel = wrap([row_LUbulk_gumbel, dashboard_gumbel], "Gumbel")
        layout_beta = wrap([row_LUbulk_beta, dashboard_beta], "Beta")

    tabs = pn.Tabs(
        layout_normal, layout_studentt, layout_expon, layout_gamma, layout_invgamma,
        layout_gumbel, layout_weibull, layout_pareto, layout_lognormal, layout_cauchy, layout_beta)

    prior_dashboard = pn.Column(md_title, tabs)

    return prior_dashboard.servable()
