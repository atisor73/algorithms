import bokeh.io
import panel as pn

# font-families worth trying: "Open Sans", "Computer Modern"
blurb_style = {'color':'#444444', 'font-size': '14px', "font-family": 'Palatino'}
frac_style = {'color':'#444444', 'font-size': '13px', "font-family": 'Palatino'}

# ****************************************************************************
width_title, width_content = 80, 515
height_story = 65
height_params = 50
height_support = 30
height_pdf = 40
height_cdf = 40
height_moments = 50
height_usage = 50
height_comments = 0


# LHS TITLES, RIGHT ALIGN
m_story = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    **Story**: """, style=blurb_style, width=width_title, height=height_story, height_policy="fixed", margin=0)
m_params = pn.pane.Markdown("""**Parameters**: """,
    style=blurb_style, width=width_title, height=height_params, height_policy="fixed", margin=0)
m_support = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;
    **Support**: """, style=blurb_style, width=width_title, height=height_support, height_policy="fixed", margin=0)
m_pdf = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    **PDF**: """, style=blurb_style, width=width_title, height=height_pdf, height_policy="fixed", margin=0)
m_cdf = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    **CDF**: """, style=blurb_style, width=width_title, height=height_cdf, height_policy="fixed", margin=0)
m_moments = pn.pane.Markdown("""&nbsp;&nbsp;
    **Moments**: """, style=blurb_style, width=width_title, height=height_moments, height_policy="fixed", margin=0)
m_usage = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    **Usage**: """,style=blurb_style, width=width_title, height=height_usage, height_policy="fixed", margin=0)
m_comments = pn.pane.Markdown("""
    **Comments**: """,style=blurb_style, width=width_title, height=height_comments, height_policy="fixed", margin=0)



# # LHS TITLES, CENTER ALIGN
# m_story = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#     **Story**: """, style=blurb_style, width=width_title, height=height_story, height_policy="fixed", margin=0)
# m_params = pn.pane.Markdown("""**Parameters**: """,
#     style=blurb_style, width=width_title, height=height_params, height_policy="fixed", margin=0)
# m_support = pn.pane.Markdown("""&nbsp;&nbsp;
#     **Support**: """, style=blurb_style, width=width_title, height=height_support, height_policy="fixed", margin=0)
# m_pdf = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#     **PDF**: """, style=blurb_style, width=width_title, height=height_pdf, height_policy="fixed", margin=0)
# m_cdf = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#     **CDF**: """, style=blurb_style, width=width_title, height=height_cdf, height_policy="fixed", margin=0)
# m_moments = pn.pane.Markdown("""&nbsp;
#     **Moments**: """, style=blurb_style, width=width_title, height=height_moments, height_policy="fixed", margin=0)
# m_usage = pn.pane.Markdown("""&nbsp;&nbsp;&nbsp;&nbsp;
#     **Usage**: """,style=blurb_style, width=width_title, height=height_usage, height_policy="fixed", margin=0)
# m_comments = pn.pane.Markdown("""
#     **Comments**: """,style=blurb_style, width=width_title, height=height_comments, height_policy="fixed", margin=0)


# RHS CONTENT
m_story_ = pn.pane.LaTeX("""
    Any quantity that emerges as the sum of a large number of subprocesses
    tends to be Normally distributed provided none of the subprocesses is very broadly distributed.<br><br>
    """, style=blurb_style, width=width_content, height=height_story, margin=0)

m_params_ = pn.pane.LaTeX("""
    $\mu$: location of peak, $(-\infty, \infty)$ <br>
    $\sigma$: width of peak, $(0, \infty$) <br><br>
    """, style=blurb_style, width=width_content, height=height_params, margin=0)

m_support_ = pn.pane.LaTeX("""
    set of real numbers $\mathbb{R}$ <br><br>
    """, style=blurb_style, width=width_content, height=height_support, margin=0)

m_pdf_ = pn.pane.LaTeX(r"""
 $f(y; \mu, \sigma) = \dfrac{1}{2\pi\sigma^2} e^{-(y-\mu)^2 / 2 \sigma^2}$ <br><br>
    """, style=frac_style, width=width_content, height=height_pdf, margin=0)

# displaystyle, dfrac, ddfrac
m_cdf_ = pn.pane.LaTeX(r"""
    $F(y; \mu, \sigma)= \dfrac{1}{2} \left[ 1 + \mathrm{erf}\left(\displaystyle\frac{x-\mu}{\sigma \sqrt{2}}\right) \right]$
    """, style=frac_style, width=width_content, height=height_cdf, margin=0)

m_moments_ = pn.pane.LaTeX("""
    mean: $\mu$  <br>
    variance: $\sigma^2$ <br><br>
    """, style=blurb_style, width=width_content, height=height_moments, margin=0)

m_usage_ = pn.pane.LaTeX("""
    SciPy: &nbsp;  <code>scipy.stats.norm(mu, sigma)</code> <br>
    Stan: &nbsp;&nbsp;&nbsp; <code>normal(mu, sigma)</code> <br>
    """, style=blurb_style, width=width_content, height=height_usage, margin=0)

m_comments_ = pn.pane.LaTeX("""
    When using the normal as a prior, note that it has <em>extremely</em> light tails,
    which can be very convenient in some settings, but overwhelming in others.<br><br>
    """, style=blurb_style, width=width_content, height=height_comments, margin=0)

m_all = pn.Column(
    pn.Row(m_story, m_story_),
    pn.Row(m_params, m_params_),
    pn.Row(m_support, m_support_),
    pn.Row(m_pdf, m_pdf_),
    pn.Row(m_cdf, m_cdf_),
    pn.Row(m_moments, m_moments_),
    pn.Row(m_usage, m_usage_),
    pn.Row(m_comments, m_comments_),
)

desc = pn.Row(pn.Spacer(width=15), pn.Column(pn.Spacer(height=25), m_all))
